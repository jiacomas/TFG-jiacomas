from dataclasses import dataclass


@dataclass
class Description:
    ID_REGISTRE: str = "id"
    TEXT_RAW: str = "text"
    TEXT_CLEAN: str = "text_clean"


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
    PDF_URL: str = "PDF"
