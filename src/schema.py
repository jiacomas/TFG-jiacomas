from dataclasses import dataclass


@dataclass
class Description:
    ID_REGISTRE = "id"
    TEXT = "text"
    YEAR = "any_publicacio"


@dataclass
class Metadata:
    ID = "ANH_ID"
    DATE = "ANU_DATA_PUBLICACIO"
    ID_REGISTRE = "ANU_NUM_REGISTRE"
    ORGANIZATION = "ORG_NOM"
    TITLE = "ANH_TITOL"
    TYPE = "Tipus Anunci"
    ODS = "ODS_NOM"
    PDF_URL = "PDF"
    FULL_TEXT = "text_final"


@dataclass
class ProcessedData:
    ID = "ANH_ID"
    FULL_TEXT = "full_text"
    ODS_LIST = "ods_list"
    TEXT_DL = "text_dl"
    TEXT_ML = "text_ml"
