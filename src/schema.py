from dataclasses import dataclass


@dataclass
class Description:
    ID_REGISTRE: str = "id"
    TEXT: str = "text"
    YEAR: str = "any_publicacio"


@dataclass
class Metadata:
    ID: str = "ANH_ID"
    DATE: str = "ANU_DATA_PUBLICACIO"
    ID_REGISTRE: str = "ANU_NUM_REGISTRE"
    ORGANIZATION: str = "ORG_NOM"
    TITLE: str = "ANH_TITOL"
    TYPE: str = "Tipus Anunci"
    ODS: list[str] = "ODS_NOM"
    PDF_URL: str = "PDF"
    FULL_TEXT: str = "text_final"


@dataclass
class ProcessedData:
    ID: str = "ANH_ID"
    FULL_TEXT: str = "full_text"
    ODS_LIST: list[str] = "ods_list"
    TEXT_DL: str = "text_dl"
    TEXT_ML: str = "text_ml"
