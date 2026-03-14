from dataclasses import dataclass


@dataclass
class Description:
    ID_REGISTRE: str = "id"
    TEXT_RAW: str = "text"
    TEXT_CLEAN: str = "text_clean"
    YEAR: str = "any_publicacio"


@dataclass
class Metadata:
    ID: str = "ANH_ID"
    DATE: str = "ANU_DATA_PUBLICACIO"
    ID_REGISTRE: str = "ANU_NUM_REGISTRE"
    ORGANIZATION: str = "ORG_NOM"
    TITLE: str = "ANH_TITOL"
    TITLE_CLEAN: str = "ANH_TITOL_CLEAN"
    TYPE: str = "Tipus Anunci"
    ODS: list[str] = "ODS_NOM"
    ODS_LIST: list[str] = "ods_list"
    PDF_URL: str = "PDF"
    FULL_TEXT: str = "text_final"
