from abc import ABC
from typing import Any, Dict


# Key text metadata attributes
TEXT_ID_ATTR = 'text_id'
TEXT_LABEL_ATTR = 'text_label'
TEXT_LABEL = 'text'


class MetaData(ABC):
    '''
    Container for managing and providing access to meta-data.
    '''

    def __init__(self, key_data: Dict[str, Any], **kwargs):
        '''
        Initialize with the mandatory or "key" data and any additional optional
        values.
        '''
        self._data = key_data.copy() if key_data is not None else dict()
        if kwargs is not None:
            self._data.update(kwargs)

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    def get_value(self, attribute: str, missing: str = None) -> Any:
        '''
        Get the value for the given attribute, or the "missing" value.
        :param attribute: The meta-data attribute whose value to get
        :param missing: The missing value
        :return: The attribute's value or the missing value.
        '''
        return self.data.get(attribute, missing)


class TextMetaData(MetaData):
    '''
    Container for text meta-data
    '''

    def __init__(
            self,
            text_id: Any,
            text_label: str = TEXT_LABEL,
            **kwargs
    ):
        super().__init__(
            {
                TEXT_ID_ATTR: text_id,
                TEXT_LABEL_ATTR: text_label,
            },
            **kwargs
        )

    @property
    def text_id(self) -> Any:
        return self.data[TEXT_ID_ATTR]

    @property
    def text_label(self) -> str:
        return self.data[TEXT_LABEL_ATTR]


class Text:
    '''
    Wrapper for a text string for analysis.
    '''

    def __init__(
            self,
            text: str,
            metadata: TextMetaData,
    ):
        self._text = text
        self._metadata = metadata

    @property
    def text(self) -> str:
        return self._text

    @property
    def text_id(self) -> Any:
        return self.metadata.text_id

    @property
    def text_label(self) -> str:
        return self.metadata.text_label

    @property
    def metadata(self) -> str:
        if self._metadata is None:
            self._metadata = TextMetaData(text_id=0)
        return self._metadata
