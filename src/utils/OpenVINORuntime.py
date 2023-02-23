import logging
from openvino.runtime import Core

from abc import abstractmethod, ABC
from pathlib import Path


class OpenVINORuntime(ABC):
    def __init__(self,
                 model: Path,
                 weights: Path,
                 output_layer_name: str,
                 score_threshold: float,
                 nms_threshold: float,
                 width: int,
                 height: int):
        model = Path(model)
        weights = Path(weights)

        if not model.is_file():
            raise FileNotFoundError(f'Model file "{model}" not found')

        if not weights.is_file():
            raise FileNotFoundError(f'Weights file "{weights}" not found')

        logging.info('Creating inference engine')
        self.__ie = Core()

        logging.info('Loading network')
        self.__model = self.__ie.read_model(model=model)
        self.__compiled_model = self.__ie.compile_model(model=self.__model, device_name='CPU')

        self.__input_layer_ir = self.__compiled_model.input(0)
        self.__output_layer_ir = self.__compiled_model.output(output_layer_name)

        self._score_threshold: float = score_threshold
        self._nms_threshold: float = nms_threshold
        self._width: int = width
        self._height: int = height

        logging.info(f'Image input height: {self._height}')
        logging.info(f'Image input width: {self._width}')

    @abstractmethod
    def _pre_processing(self, image):
        raise NotImplementedError

    def __infer(self, data):
        return self.__compiled_model([data])[self.__output_layer_ir]

    @abstractmethod
    def _post_processing(self, output, image):
        raise NotImplementedError

    def predict(self, image):
        prepared_input = self._pre_processing(image=image)
        output = self.__infer(data=prepared_input)
        return self._post_processing(output=output, image=image)
