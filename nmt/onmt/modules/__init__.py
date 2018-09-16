from onmt.modules.UtilClass import LayerNorm, Elementwise
from onmt.modules.Gate import context_gate_factory, ContextGate
from onmt.modules.GlobalAttention import GlobalAttention
from onmt.modules.CopyGenerator import CopyGenerator, CopyGeneratorLossCompute
from onmt.modules.StructuredAttention import MatrixTree
from onmt.modules.Transformer import \
   TransformerEncoder, TransformerDecoder, PositionwiseFeedForward
from onmt.modules.MultiHeadedAttn import MultiHeadedAttention
from onmt.modules.StackedRNN import StackedLSTM, StackedGRU
from onmt.modules.Embeddings import Embeddings, PositionalEncoding
from onmt.modules.WeightNorm import WeightNormConv2d

from onmt.Models import EncoderBase, MeanEncoder, StdRNNDecoder, \
    RNNDecoderBase, InputFeedRNNDecoder, RNNEncoder, NMTModel

from onmt.modules.SRU import check_sru_requirement
can_use_sru = check_sru_requirement()
if can_use_sru:
    from onmt.modules.SRU import SRU


# For flake8 compatibility.
__all__ = [EncoderBase, MeanEncoder, RNNDecoderBase, InputFeedRNNDecoder,
           RNNEncoder, NMTModel,
           StdRNNDecoder, ContextGate, GlobalAttention, 
           PositionwiseFeedForward, PositionalEncoding,
           CopyGenerator, MultiHeadedAttention,
           LayerNorm,
           TransformerEncoder, TransformerDecoder, Embeddings, Elementwise,
           MatrixTree, 
           StackedLSTM, StackedGRU,
           context_gate_factory, CopyGeneratorLossCompute]

if can_use_sru:
    __all__.extend([SRU, check_sru_requirement])
