from typing import Any, Union, Tuple, List, Iterable, Dict, Sequence, Optional, NamedTuple
import os
import json
import torch
import logging
import operator
from functools import reduce
from vncorenlp import VnCoreNLP
from collections import UserDict
from collections import namedtuple
from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE

logger = logging.getLogger(__name__)
CONFIG_FILE = 'photokenizer_config.json'
VOCAB_FILE = 'dict.txt'
BPECODE_FILE = 'bpe.codes'

LARGE_INTEGER = int(1e20)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER

# Define type aliases and NamedTuples
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]

class BatchEncoding(UserDict):
    """ BatchEncoding hold the output of the encode and batch_encode methods (tokens, attention_masks, etc).
        This class is derived from a python Dictionary and can be used as a dictionnary.
        In addition, this class expose utility methods to map from word/char space to token space.

        Args:
            data (:obj:`dict`): Dictionary of lists/arrays returned by the encode/batch_encode methods ('input_ids', 'attention_mask'...)
            encoding (:obj:`EncodingFast`, :obj:`list(EncodingFast)`, `optional`, defaults to :obj:`None`):
                If the tokenizer is a fast tokenizer which outputs additional informations like mapping from word/char space to token space
                the `EncodingFast` instance or list of instance (for batches) hold these informations.

    """

    def __init__(self, data: Dict[str, Any]):
        super().__init__(data)

    def __getitem__(self, item: Union[int, str]):
        """ If the key is a string, get the value of the dict associated to `key` ('input_ids', 'attention_mask'...)
            If the key is an integer, get the EncodingFast for batch item with index `key`
        """
        if isinstance(item, str):
            return self.data[item]
        else:
            raise KeyError(
                "Indexing with integers (to access backend Encoding for a given batch index) "
                "is not available when using Python based tokenizers"
            )

    def __getattr__(self, item: str):
        return self.data[item]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()


class PhoTokenizer(object):
    """ Process input for PhoBERT

        Segment words and then convert them into ids
    """

    model_input_names: List[str] = ["attention_mask"]

    model_max_length: int = 258
    padding_side: str = "right"
    _pad_token_type_id: int = 0
    

    @property
    def pad_token_type_id(self):
        """ Id of the padding token type in the vocabulary."""
        return self._pad_token_type_id

    @property
    def bos_token_id(self):
        """ Id of the beginning of sentence token in the vocabulary."""
        return self.vocab.bos_index

    @property
    def eos_token_id(self):
        """ Id of the end of sentence token in the vocabulary."""
        return self.vocab.eos_index

    @property
    def unk_token_id(self):
        """ Id of the unknown token in the vocabulary."""
        return self.vocab.unk_index

    @property
    def sep_token_id(self):
        """ Id of the separation token in the vocabulary. E.g. separate context and query in an input sequence."""
        return self.vocab.eos_index

    @property
    def pad_token_id(self):
        """ Id of the padding token in the vocabulary."""
        return self.vocab.pad_index

    @property
    def cls_token_id(self):
        """ Id of the classification token in the vocabulary. E.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the model."""
        return self.vocab.bos_index

    def __init__(self, bpe_path: str, vncorenlp_path: str, do_lower_case: bool = False):
        bpe_codes_path = os.path.join(bpe_path, BPECODE_FILE)
        vocab_file_path = os.path.join(bpe_path, VOCAB_FILE)
        
        if not os.path.isfile(bpe_codes_path):
            raise EnvironmentError(f"{BPECODE_FILE} not found in {bpe_path}")
            
        if not os.path.isfile(vocab_file_path):
            raise EnvironmentError(f"{VOCAB_FILE} not found in {bpe_path}")

        self.do_lower_case = do_lower_case
        
        BPEConfig = namedtuple('BPEConfig', 'vncorenlp bpe_codes vocab')

        self.pho_config = BPEConfig(vncorenlp=vncorenlp_path, bpe_codes=bpe_codes_path, vocab=vocab_file_path)
        self.rdrsegmenter = VnCoreNLP(self.pho_config.vncorenlp, annotators="wseg", max_heap_size='-Xmx1g')
        self.bpe = fastBPE(self.pho_config)
        self.vocab = Dictionary()
        self.vocab.add_from_file(self.pho_config.vocab)

    @staticmethod
    def load(model_path: str, **kwargs):
        config_path = os.path.join(model_path, CONFIG_FILE)
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as fIn:
                config = json.load(fIn)
        elif len(kwargs) == 0:
            raise EnvironmentError("{CONFIG_FILE} not found. Please initialize model instead of using load method.")
        
        config.update(kwargs)
        
        return PhoTokenizer(**config)

    def save(self, output_path: str):
        with open(os.path.join(output_path, CONFIG_FILE), 'w') as fOut:
            json.dump({'bpe_path': self.pho_config.vocab.replace(VOCAB_FILE, ''), 'vncorenlp_path': self.pho_config.vncorenlp, 'do_lower_case': self.do_lower_case}, fOut)

    def segment(self, text: str) -> str:
        ''' Segment words in text and then flat the list '''
        segmented_word = self.rdrsegmenter.tokenize(text)
        return ' '.join(reduce(operator.concat, segmented_word))
        
    def convert_tokens_to_ids(self, text: str) -> List[int]:
        return self.vocab.encode_line(text, append_eos=False, add_if_not_exist=False).tolist()

    def tokenize(self, text: str) -> str:
        if self.do_lower_case:
            map(str.lower, text)

        sent = self.segment(text)
        return self.bpe.encode(sent)
    
    # def encode(self, text: str) -> List[int]:
    #     return self.convert_tokens_to_ids(self.tokenize(text))
         

    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        pad_to_max_length: bool = False,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_lengths: bool = False,
    ) -> BatchEncoding:
        """ Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model.
        It adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            max_length: maximum length of the returned list. Will truncate by taking into account the special tokens.
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            stride: window stride for overflowing tokens. Can be useful to remove edge effect when using sequential
                list of inputs. The overflowing token will contains a part of the previous window of tokens.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            pad_to_max_length: if set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the model's max length.
                The tokenizer padding sides are handled by the following strings:
                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
                Defaults to False: no padding.
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            return_token_type_ids: (optional) Set to False to avoid returning token_type_ids (default: set to model specifics).
            return_attention_mask: (optional) Set to False to avoid returning attention mask (default: set to model specifics)
            return_overflowing_tokens: (optional) Set to True to return overflowing token information (default False).
            return_special_tokens_mask: (optional) Set to True to return special tokens mask information (default False).
            return_lengths (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If set the resulting dictionary will include the length of each encoded inputs

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[int],
                    token_type_ids: list[int] if return_token_type_ids is True (default)
                    overflowing_tokens: list[int] if a ``max_length`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: int if a ``max_length`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
                    length: int if return_lengths is True
                }

            With the fields:
                - ``input_ids``: list of token ids to be fed to a model
                - ``token_type_ids``: list of token type ids to be fed to a model

                - ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
                - ``num_truncated_tokens``: number of overflowing tokens a ``max_length`` is specified
                - ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
                    tokens and 1 specifying sequence tokens.
                - ``length``: this is the length of ``input_ids``
        """
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Truncation: Handle max sequence length
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)
        if max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )
            if return_overflowing_tokens:
                encoded_inputs["overflowing_tokens"] = overflowing_tokens
                encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([1] * len(pair_ids) if pair else [])

        # Build output dictionnary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        assert max_length is None or len(encoded_inputs["input_ids"]) <= max_length
        if max_length is None and len(encoded_inputs["input_ids"]) > self.model_max_length:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum sequence length "
                "for this model ({} > {}). Running this sequence through the model will result in "
                "indexing errors".format(len(ids), self.model_max_length)
            )

        # Padding
        needs_to_be_padded = pad_to_max_length and (
            max_length
            and len(encoded_inputs["input_ids"]) < max_length
            or max_length is None
            and len(encoded_inputs["input_ids"]) < self.model_max_length
            and self.model_max_length <= LARGE_INTEGER
        )

        if pad_to_max_length and max_length is None and self.model_max_length > LARGE_INTEGER:
            logger.warning(
                "Sequence can't be padded as no maximum length is specified and the model maximum length is too high."
            )

        if needs_to_be_padded:
            difference = (max_length if max_length is not None else self.model_max_length) - len(
                encoded_inputs["input_ids"]
            )
            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"]) + [0] * difference
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [self.pad_token_id] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [1] * len(encoded_inputs["input_ids"])
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs["input_ids"] = [self.pad_token_id] * difference + encoded_inputs["input_ids"]
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        else:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])

        if return_lengths:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        # Prepare model inputs as tensors if asked
        if return_tensors == "pt":
            encoded_inputs["input_ids"] = torch.tensor([encoded_inputs["input_ids"]])

            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = torch.tensor([encoded_inputs["token_type_ids"]])

            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = torch.tensor([encoded_inputs["attention_mask"]])
        elif return_tensors is not None:
            logger.warning(
                "Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(
                    return_tensors
                )
            )

        return BatchEncoding(encoded_inputs)

    def num_special_tokens_to_add(self, pair=False):
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        Note:
            This encodes inputs and checks the number of added tokens, and is therefore not efficient. Do not put this
            inside your training loop.

        Args:
            pair: Returns the number of added tokens in the case of a sequence pair if set to True, returns the
                number of added tokens in the case of a single sequence if set to False.

        Returns:
            Number of tokens added to sequences
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A RoBERTa sequence has the following format:

        - single sequence: ``<s> X </s>``
        - pair of sequences: ``<s> A </s></s> B </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep


    def truncate_sequences(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: str = "longest_first",
        stride: int = 0,
    ) -> Tuple[List[int], List[int], List[int]]:
        """ Truncates a sequence pair in place to the maximum length.

        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            num_tokens_to_remove (:obj:`int`, `optional`, defaults to ``0``):
                number of tokens to remove using the truncation strategy
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences).
                    Overflowing tokens only contains overflow from the first sequence.
                - 'only_first': Only truncate the first sequence. raise an error if the first sequence is shorter or equal to than num_tokens_to_remove.
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            stride (:obj:`int`, `optional`, defaults to ``0``):
                If set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
        """
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if truncation_strategy == "longest_first":
            overflowing_tokens = []
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    overflowing_tokens = [ids[-1]] + overflowing_tokens
                    ids = ids[:-1]
                else:
                    pair_ids = pair_ids[:-1]
            window_len = min(len(ids), stride)
            if window_len > 0:
                overflowing_tokens = ids[-window_len:] + overflowing_tokens
        elif truncation_strategy == "only_first":
            assert len(ids) > num_tokens_to_remove
            window_len = min(len(ids), stride + num_tokens_to_remove)
            overflowing_tokens = ids[-window_len:]
            ids = ids[:-num_tokens_to_remove]
        elif truncation_strategy == "only_second":
            assert pair_ids is not None and len(pair_ids) > num_tokens_to_remove
            window_len = min(len(pair_ids), stride + num_tokens_to_remove)
            overflowing_tokens = pair_ids[-window_len:]
            pair_ids = pair_ids[:-num_tokens_to_remove]
        elif truncation_strategy == "do_not_truncate":
            raise ValueError("Input sequence are too long for max_length. Please select a truncation strategy.")
        else:
            raise ValueError(
                "Truncation_strategy should be selected in ['longest_first', 'only_first', 'only_second', 'do_not_truncate']"
            )
        return (ids, pair_ids, overflowing_tokens)

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        RoBERTa does not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of zeros.

        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 0 for a special token, 1 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        pad_to_max_length: bool = False,
        is_pretokenized: bool = False,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_masks: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_masks: bool = False,
        return_offsets_mapping: bool = False,
        return_lengths: bool = False,
        **kwargs
    ) -> BatchEncoding:
        """
        Returns a dictionary containing the encoded sequence or sequence pair and additional information:
        the mask for sequence classification and the overflowing elements if a ``max_length`` is specified.

        Args:
            batch_text_or_text_pairs (:obj:`List[str]`,  :obj:`List[Tuple[str, str]]`,
                                      :obj:`List[List[str]]`,  :obj:`List[Tuple[List[str], List[str]]]`,
                                      and for not-fast tokenizers, also:
                                      :obj:`List[List[int]]`,  :obj:`List[Tuple[List[int], List[int]]]`):
                Batch of sequences or pair of sequences to be encoded.
                This can be a list of string/string-sequences/int-sequences or a list of pair of
                string/string-sequences/int-sequence (see details in encode_plus)
            add_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            max_length (:obj:`int`, `optional`, defaults to :obj:`None`):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
            stride (:obj:`int`, `optional`, defaults to ``0``):
                If set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
            truncation_strategy (:obj:`str`, `optional`, defaults to `longest_first`):
                String selected in the following options:

                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                  starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            pad_to_max_length (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If set to True, the returned sequences will be padded according to the model's padding side and
                padding index, up to their max length. If no max length is specified, the padding is done up to the
                model's max length. The tokenizer padding sides are handled by the class attribute `padding_side`
                which can be set to the following strings:

                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
                Defaults to False: no padding.
            is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
                Set to True to indicate the input is already tokenized
            return_tensors (:obj:`str`, `optional`, defaults to :obj:`None`):
                Can be set to 'tf' or 'pt' to return respectively TensorFlow :obj:`tf.constant`
                or PyTorch :obj:`torch.Tensor` instead of a list of python integers.
            return_token_type_ids (:obj:`bool`, `optional`, defaults to :obj:`None`):
                Whether to return token type IDs. If left to the default, will return the token type IDs according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are token type IDs? <../glossary.html#token-type-ids>`_
            return_attention_masks (:obj:`bool`, `optional`, defaults to :obj:`none`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are attention masks? <../glossary.html#attention-mask>`__
            return_overflowing_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return overflowing token information (default False).
            return_special_tokens_masks (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return special tokens mask information (default False).
            return_offsets_mapping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True to return (char_start, char_end) for each token (default False).
                If using Python's tokenizer, this method will raise NotImplementedError. This one is only available on
                Rust-based tokenizers inheriting from PreTrainedTokenizerFast.
            return_lengths (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If set the resulting dictionary will include the length of each encoded inputs
            **kwargs: passed to the `self.tokenize()` method

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[List[int]],
                    token_type_ids: list[List[int]] if return_token_type_ids is True (default)
                    attention_mask: list[List[int]] if return_attention_mask is True (default)
                    overflowing_tokens: list[List[int]] if a ``max_length`` is specified and return_overflowing_tokens is True
                    num_truncated_tokens: List[int] if a ``max_length`` is specified and return_overflowing_tokens is True
                    special_tokens_mask: list[List[int]] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
                }

            With the fields:

            - ``input_ids``: list of token ids to be fed to a model
            - ``token_type_ids``: list of token type ids to be fed to a model
            - ``attention_mask``: list of indices specifying which tokens should be attended to by the model
            - ``overflowing_tokens``: list of overflowing tokens if a max length is specified.
            - ``num_truncated_tokens``: number of overflowing tokens a ``max_length`` is specified
            - ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
              tokens and 1 specifying sequence tokens.
        """

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        # Throw an error if we can pad because there is no padding token
        if pad_to_max_length and self.pad_token_id is None:
            raise ValueError(
                "Unable to set proper padding strategy as the tokenizer does not have a padding token. In this case please set the `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via the function add_special_tokens if you want to use a padding strategy"
            )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if isinstance(ids_or_pair_ids, (list, tuple)) and len(ids_or_pair_ids) == 2 and not is_pretokenized:
                ids, pair_ids = ids_or_pair_ids
            else:
                ids, pair_ids = ids_or_pair_ids, None

            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        if max_length is None and pad_to_max_length:

            def total_sequence_length(input_pairs):
                first_ids, second_ids = input_pairs
                return len(first_ids) + (
                    self.num_special_tokens_to_add()
                    if second_ids is None
                    else (len(second_ids) + self.num_special_tokens_to_add(pair=True))
                )

            max_length = max([total_sequence_length(ids) for ids in input_ids])

        batch_outputs = {}
        for first_ids, second_ids in input_ids:
            # Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by
            # the model. It adds special tokens, truncates sequences if overflowing while taking into account
            # the special tokens and manages a window stride for overflowing tokens
            outputs = self.prepare_for_model(
                first_ids,
                pair_ids=second_ids,
                max_length=max_length,
                pad_to_max_length=pad_to_max_length,
                add_special_tokens=add_special_tokens,
                stride=stride,
                truncation_strategy=truncation_strategy,
                return_attention_mask=return_attention_masks,
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_masks,
                return_lengths=return_lengths,
                return_tensors=None,  # We will convert the whole batch to tensors at the end
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        if return_tensors is not None:

            self.convert_to_tensors_(batch_outputs, return_tensors)
        return BatchEncoding(batch_outputs)

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]]
    ) -> Union[int, List[int]]:
        """ Converts a single index or a sequence of indices (integers) in a token "
            (resp.) a sequence of tokens (str), using the vocabulary and added tokens.
            Args:
                skip_special_tokens: Don't decode special tokens (self.all_special_tokens). Default: False
        """
        if isinstance(ids, int):
            return self.vocab[ids]

        tokens = []
        for index in ids:
            index = int(index)
            tokens.append(self.vocab[index])
        return tokens

    def save_pretrained(self, *args, **kwargs):
        return ('', '', '', '')