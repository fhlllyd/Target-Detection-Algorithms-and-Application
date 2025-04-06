# YOLOv5 🚀 от Ultralytics, лицензия GPL-3.0
""" Утилиты для взаимодействия с сервером инференса Triton
"""

import typing
from urllib.parse import urlparse

import torch


class TritonRemoteModel:
    """ Обертка над моделью, предоставляемой сервером инференса Triton. Она может
    быть настроена для общения через GRPC или HTTP. Принимает ввод в виде Torch Tensors
    и возвращает их в качестве вывода.
    """

    def __init__(self, url: str):
        """
        Ключевые аргументы:
        url: Полностью квалифицированный адрес сервера Triton - например, grpc://localhost:8000
        """

        parsed_url = urlparse(url)
        if parsed_url.scheme == "grpc":
            from tritonclient.grpc import InferenceServerClient, InferInput

            self.client = InferenceServerClient(parsed_url.netloc)  # Клиент GRPC Triton
            model_repository = self.client.get_model_repository_index()
            self.model_name = model_repository.models[0].name
            self.metadata = self.client.get_model_metadata(self.model_name, as_json=True)

            def create_input_placeholders() -> typing.List[InferInput]:
                return [
                    InferInput(i['name'], [int(s) for s in i["shape"]], i['datatype']) for i in self.metadata['inputs']]

        else:
            from tritonclient.http import InferenceServerClient, InferInput

            self.client = InferenceServerClient(parsed_url.netloc)  # Клиент HTTP Triton
            model_repository = self.client.get_model_repository_index()
            self.model_name = model_repository[0]['name']
            self.metadata = self.client.get_model_metadata(self.model_name)

            def create_input_placeholders() -> typing.List[InferInput]:
                return [
                    InferInput(i['name'], [int(s) for s in i["shape"]], i['datatype']) for i in self.metadata['inputs']]

        self._create_input_placeholders_fn = create_input_placeholders

    @property
    def runtime(self):
        """Возвращает среду выполнения модели"""
        return self.metadata.get("backend", self.metadata.get("platform"))

    def __call__(self, *args, **kwargs) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]]:
        """ Вызывает модель. Параметры могут быть предоставлены через args или kwargs.
        args, если предоставлены, предполагаются соответствовать порядку ввода модели.
        kwargs сопоставляются с именами входов модели.
        """
        inputs = self._create_inputs(*args, **kwargs)
        response = self.client.infer(model_name=self.model_name, inputs=inputs)
        result = []
        for output in self.metadata['outputs']:
            tensor = torch.as_tensor(response.as_numpy(output['name']))
            result.append(tensor)
        return result[0] if len(result) == 1 else result

    def _create_inputs(self, *args, **kwargs):
        args_len, kwargs_len = len(args), len(kwargs)
        if not args_len and not kwargs_len:
            raise RuntimeError("Не предоставлены входные данные.")
        if args_len and kwargs_len:
            raise RuntimeError("Нельзя указывать args и kwargs одновременно")

        placeholders = self._create_input_placeholders_fn()
        if args_len:
            if args_len != len(placeholders):
                raise RuntimeError(f"Ожидалось {len(placeholders)} входных данных, получено {args_len}.")
            for input, value in zip(placeholders, args):
                input.set_data_from_numpy(value.cpu().numpy())
        else:
            for input in placeholders:
                value = kwargs[input.name]
                input.set_data_from_numpy(value.cpu().numpy())
        return placeholders