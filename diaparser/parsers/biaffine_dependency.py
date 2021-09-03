# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn

from ..models import BiaffineDependencyModel
from .parser import Parser
from ..utils import Config, Dataset, Embedding
from ..utils.common import bos, pad, unk
from ..utils.field import Field, SubwordField, BertField
from ..utils.fn import ispunct
from ..utils.logging import get_logger, progress_bar
from ..utils.metric import AttachmentMetric
from ..utils.transform import CoNLL
from tokenizer.tokenizer import Tokenizer

logger = get_logger(__name__)


COPTIC_TAGS = ['UNKNOWN', 'ADV', 'CFOC_PPERS', 'PINT', 'PPERS', 'V_PPERO', 'ART', 'AAOR', 'FUT', 'CONJ', 'VIMP',
               'CPRET', 'COP', 'ANEGPST', 'PPERI', 'NPROP', 'APST', 'PUNCT', 'PREP_PPERO', 'N_PPERO', 'PPOS', 'VSTAT',
               'ANEGOPT', 'FM', 'APST_PPERS', 'ACONJ', 'ANEGPST_PPERS', 'NUM', 'CPRET_PPERS', 'ANY', 'CREL', 'NEG',
               'ACOND_PPERS', 'PTC', 'ACOND', 'CFOC', 'ACAUS', 'APREC', 'IMOD', 'ALIM', 'ACONJ_PPERS', 'AOPT', 'VBD',
               'IMOD_PPERO', 'CCIRC_PPERS', 'EXIST', 'CREL_PPERS', 'V', 'N', 'AOPT_PPERS', 'PDEM', 'PPERO', 'ANEGJUS',
               'PREP', 'AFUTCONJ', 'AJUS', 'ANEGAOR', 'CCIRC']


def read_coptic_embeds():
    with open('data/coptic_50d.vec') as f:
        lines = f.readlines()

    m = {}
    for line in lines:
        word, *embs = line.strip().split(" ")
        if len(embs) != 50:
            print("Skipping word in coptic embeddings: less than 50 d")
            continue
        embs = torch.tensor([float(d) for d in embs])
        m[word] = embs

    unk_emb = torch.vstack([v for v in m.values()]).mean(dim=0)
    return m, unk_emb


EMBEDS, UNK_EMB = read_coptic_embeds()


def read_words(sent):
    annos = sorted([a for a in sent.annotations.items() if a[0] >= 0], key=lambda a:a[0])
    words = []
    for _, a in annos:
        cols = a.split('\t')
        w = cols[1]
        e = EMBEDS[w] if w in EMBEDS else UNK_EMB
        words.append(e)

    return torch.vstack(words)


def read_tags(sent):
    annos = sorted([a for a in sent.annotations.items() if a[0] >= 0], key=lambda a: a[0])
    tags = []
    for _, a in annos:
        cols = a.split('\t')
        tags.append(COPTIC_TAGS.index(cols[4]))
    return torch.tensor(tags)


def pad_sequence(tensors, pad_tensor, embs=False):
    maxlen = max(t.shape[0] for t in tensors)

    new_tensors = []
    for i, tensor in enumerate(tensors):
        diff = maxlen - tensor.shape[0]
        pad = [pad_tensor] * diff
        if embs:
            pad = [p.unsqueeze(0) for p in pad]
            padded_tensor = torch.cat([tensor] + pad, dim=0)
        else:
            padded_tensor = torch.hstack([tensor] + pad)
        new_tensors.append(padded_tensor)
    if embs:
        return torch.cat([t.unsqueeze(0) for t in new_tensors], dim=0)
    else:
        return torch.vstack(new_tensors)


def tags_and_static_embs(loader, i):
    sentence_indices = loader.batch_sampler.buckets[i]
    sentences = [s for i, s in enumerate(loader.dataset.sentences) if i in sentence_indices]
    tags = pad_sequence([read_tags(s) for s in sentences], torch.tensor(0))
    static_embs = pad_sequence([read_words(s) for s in sentences], torch.zeros(50), embs=True)
    tags = torch.cat((torch.zeros((tags.shape[0], 1)), tags), dim=1).int()
    static_embs = torch.cat((torch.zeros((static_embs.shape[0], 1, 50)), static_embs), dim=1)
    return tags, static_embs


class BiaffineDependencyParser(Parser):
    r"""
    The implementation of Biaffine Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning. 2017.
          `Deep Biaffine Attention for Neural Dependency Parsing`_.

    .. _Deep Biaffine Attention for Neural Dependency Parsing:
        https://openreview.net/forum?id=Hk95PK9le
    """

    MODEL = BiaffineDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.args.feat in ('char', 'bert'):
            self.WORD, self.FEAT = self.transform.FORM
        else:
            self.WORD, self.FEAT = self.transform.FORM, self.transform.CPOS
        self.ARC, self.REL = self.transform.HEAD, self.transform.DEPREL
        self.puncts = torch.tensor([i
                                    for s, i in self.WORD.vocab.stoi.items()
                                    if ispunct(s)]).to(self.args.device)

    def train(self, train, dev, test, buckets=32, batch_size=5000,
              punct=False, tree=False, proj=False, verbose=True, **kwargs):
        r"""
        Args:
            train/dev/test (list[list] or str):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            punct (bool):
                If ``False``, ignores the punctuations during evaluation. Default: ``False``.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for training.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000,
                 punct=False, tree=True, proj=False, partial=False, verbose=True, **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            punct (bool):
                If ``False``, ignores the punctuations during evaluation. Default: ``False``.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, buckets=8, batch_size=5000,
                prob=False, tree=True, proj=False, verbose=False, **kwargs):
        r"""
        Args:
            data (list[list] or str):
                The data for prediction, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for prediction.

        Returns:
            A :class:`~diaparser.utils.Dataset` object that stores the predicted results.
        """

        return super().predict(**Config().update(locals()))

    def _train(self, loader):
        self.model.train()

        bar, metric = progress_bar(loader), AttachmentMetric()

        for i, (words, feats, arcs, rels) in enumerate(bar):
            self.optimizer.zero_grad()
            tags, static_embs = tags_and_static_embs(loader, i)

            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.model(words, feats, tags, static_embs)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, self.args.partial)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            metric(arc_preds, rel_preds, arcs, rels, mask)
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}")

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, AttachmentMetric()

        for i, (words, feats, arcs, rels) in enumerate(loader):
            tags, static_embs = tags_and_static_embs(loader, i)
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.model(words, feats, tags, static_embs)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask, self.args.partial)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask,
                                                     self.args.tree,
                                                     self.args.proj)
            if self.args.partial:
                mask &= arcs.ge(0)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            total_loss += loss.item()
            metric(arc_preds, rel_preds, arcs, rels, mask)
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {}
        arcs, rels, probs = [], [], []
        for words, feats in progress_bar(loader):
            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc, s_rel = self.model(words, feats)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask,
                                                     self.args.tree,
                                                     self.args.proj)
            arcs.extend(arc_preds[mask].split(lens))
            rels.extend(rel_preds[mask].split(lens))
            if self.args.prob:
                arc_probs = s_arc.softmax(-1)
                probs.extend([prob[1:i+1, :i+1].cpu() for i, prob in zip(lens, arc_probs.unbind())])
        arcs = [seq.tolist() for seq in arcs]
        rels = [self.REL.vocab[seq.tolist()] for seq in rels]
        preds = {'arcs': arcs, 'rels': rels}
        if self.args.prob:
            preds['probs'] = probs

        return preds

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        WORD = Field('words', pad="[PAD]", unk="[UNK]", bos="[CLS]", lower=False)
        if args.feat == 'char':
            FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos, fix_len=args.fix_len)
        elif args.feat == 'bert':
            tokenizer = BertField.tokenizer(args.bert)

            args.max_len = min(args.max_len or tokenizer.model_max_length, tokenizer.model_max_length)
            FEAT = BertField('bert', tokenizer, fix_len=args.fix_len)
            WORD.bos = FEAT.bos  # ensure representations have the same length
        else:
            FEAT = Field('tags', bos=bos)
        ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
        REL = Field('rels', bos=bos)
        if args.feat in ('char', 'bert'):
            transform = CoNLL(FORM=(WORD, FEAT), HEAD=ARC, DEPREL=REL)
        else:
            transform = CoNLL(FORM=WORD, CPOS=FEAT, HEAD=ARC, DEPREL=REL)

        train = Dataset(transform, args.train)
        WORD.build(train, args.min_freq, (Embedding.load(args.embed, args.unk) if args.embed else None))
        FEAT.build(train)
        REL.build(train)
        # set parameters from data:
        args.update({
            'n_words': WORD.vocab.n_init,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'n_feats': len(FEAT.vocab),
            'n_rels': len(REL.vocab),
            'feat_pad_index': FEAT.pad_index,
        })

        logger.info("Features:")
        logger.info(f"   {WORD}")
        logger.info(f"   {FEAT}\n   {ARC}\n   {REL}")

        model = cls.MODEL(**args)
        model.load_pretrained(WORD.embed).to(args.device)
        return cls(args, model, transform)
