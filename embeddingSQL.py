
# embeddingSQL.py — build SQL for generating text embeddings in Teradata Vantage
#
# Supports both:
#   • IVSM pipeline: ivsm.tokenizer_encode -> ivsm.IVSM_score -> TABLE(ivsm.vector_to_columns)
#   • BYOM pipeline: ONNXEmbeddings (single-function approach)
#
# Usage (example):
#     import time
#     import teradataml as tdml
#     import embeddingSQL as sqlGen
#
#     start = time.time()
#     sqlSourceData = (
#         "select doc_id, text_id, txt "
#         "from man_doc_contents mdc "
#         "join man_source_docs msd on (mdc.doc_id = msd.id) "
#         "where status_cd = 1"
#     )
#     table_name = "man_doc_embeddings_384"
#
#     # Detect table
#     createTable = False
#     try:
#         _ = tdml.DataFrame(table_name)
#     except Exception:
#         createTable = True
#
#     beginFragment = f"create table {table_name} as (" if createTable else f"insert into {table_name} "
#     endFragment   = " ) with DATA" if createTable else ""
#
#     # IVSM path (set vector_length to your model dim, 384 for bge-small-en-v1.5)
#     embedSQL = f"""
#     {beginFragment}
#     {sqlGen.getSQL_IVSM(model_id='bge-small-en-v1.5',
#                         source_sql=sqlSourceData,
#                         vector_length=384,
#                         preserve_txt=True,
#                         ivsm_schema='ivsm',
#                         model_table='embeddings_models',
#                         tokenizer_table='embeddings_tokenizers')}
#     {endFragment}
#     """
#     tdml.execute_sql(embedSQL)
#     print("Elapsed min:", (time.time() - start) / 60)

from typing import Optional

# ------------------------------
# IVSM builder
# ------------------------------
def getSQL_IVSM(
    model_id: str,
    source_sql: str,
    vector_length: int = 384,
    preserve_txt: bool = True,
    ivsm_schema: str = "ivsm",
    model_table: str = "embeddings_models",
    tokenizer_table: str = "embeddings_tokenizers",
) -> str:
    """
    Build IVSM SQL to generate embeddings columns emb_0..emb_{vector_length-1}.

    Parameters
    ----------
    model_id : str
        The model identifier (e.g., 'bge-small-en-v1.5').
    source_sql : str
        SQL that yields doc_id, text_id, txt columns to embed.
    vector_length : int
        Embedding dimension for the model; default 384.
    preserve_txt : bool
        If True, include 'txt' in ColumnsToPreserve (doubles storage but convenient for testing).
    ivsm_schema : str
        Schema where IVSM functions are registered (commonly 'ivsm').
    model_table : str
        Table containing ONNX model BLOB (columns: model_id, model).
    tokenizer_table : str
        Table containing tokenizer JSON/BLOB (columns: model_id, model as tokenizer).
    """

    preserve_cols = "'doc_id','text_id'" + (",'txt'" if preserve_txt else "")
    txt_col = "txt," if preserve_txt else ""

    # 1) Tokenize/encode
    sqlPart_01 = f"""
    SELECT
        doc_id,
        text_id,
        {txt_col}
        IDS AS input_ids,
        attention_mask
    FROM {ivsm_schema}.tokenizer_encode(
        ON ({source_sql})
        ON (SELECT model AS tokenizer FROM {tokenizer_table} WHERE model_id = '{model_id}') DIMENSION
        USING
            ColumnsToPreserve({preserve_cols})
            OutputFields('IDS','ATTENTION_MASK')
            MaxLength(1024)
            PadToMaxLength('True')
            TokenDataType('INT64')
    ) AS enc
    """

    # 2) Score to get embeddings tensor
    sqlPart_02 = f"""
    SELECT *
    FROM {ivsm_schema}.IVSM_score(
        ON ({sqlPart_01})
        ON (SELECT * FROM {model_table} WHERE model_id = '{model_id}') DIMENSION
        USING
            ColumnsToPreserve({preserve_cols})
            ModelType('ONNX')
            BinaryInputFields('input_ids','attention_mask')
            BinaryOutputFields('sentence_embedding')
            Caching('inquery')
    ) AS scored
    """

    # 3) Expand tensor to float columns (emb_0..emb_{vector_length-1})
    #    Use TABLE(...) form to avoid the parser error near 'vector_to_columns ('.
    embeddingSQL = f"""
    SELECT *
    FROM TABLE (
        {ivsm_schema}.vector_to_columns(
            ON ({sqlPart_02})
            USING
                ColumnsToPreserve({preserve_cols})
                VectorDataType('FLOAT32')
                VectorLength({vector_length})
                OutputColumnPrefix('emb_')
                InputColumnName('sentence_embedding')
        )
    ) AS vc
    """

    return embeddingSQL


# ------------------------------
# BYOM builder (ONNXEmbeddings)
# ------------------------------
def getSQL_BYOM(
    model_id: str,
    source_sql: str,
    vector_length: int = 384,
    preserve_txt: bool = True,
    byom_schema: Optional[str] = "mldb",
    model_table: str = "embeddings_models",
    tokenizer_table: str = "embeddings_tokenizers",
) -> str:
    """
    Build BYOM SQL using ONNXEmbeddings to output FLOAT32({vector_length}) columns emb_0..emb_{vector_length-1}.

    Parameters
    ----------
    model_id : str
        The model identifier (e.g., 'bge-small-en-v1.5').
    source_sql : str
        SQL that yields doc_id, text_id, txt columns to embed.
    vector_length : int
        Embedding dimension for the model; default 384.
    preserve_txt : bool
        If True, include 'txt' in Accumulate list (doubles storage but convenient for testing).
    byom_schema : Optional[str]
        Schema where ONNXEmbeddings UDF is cataloged; commonly 'mldb'. If None or '', call without schema prefix.
    model_table : str
        Table containing ONNX model BLOB (columns: model_id, model).
    tokenizer_table : str
        Table containing tokenizer JSON (columns: model_id, tokenizer).
    """

    accumulate = "'doc_id','text_id'" + (",'txt'" if preserve_txt else "")
    schema_prefix = f"{byom_schema}." if byom_schema else ""

    embeddingSQL = f"""
    SELECT *
    FROM {schema_prefix}ONNXEmbeddings(
        ON ({source_sql}) AS InputTable
        ON (SELECT model_id, model FROM {model_table} WHERE model_id = '{model_id}') DIMENSION
        ON (SELECT tokenizer FROM {tokenizer_table} WHERE model_id = '{model_id}') DIMENSION
               USING
            Accumulate({accumulate})
            ModelOutputTensor('sentence_embedding')
            OutputFormat('FLOAT32({vector_length})')
    ) AS sqlmr
    """