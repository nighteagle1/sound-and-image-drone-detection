class DroneError(Exception):
    """Базовая ошибка приложения; .message показывается пользователю на русском."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ModelNotFoundError(DroneError):
    pass


class SourceError(DroneError):
    pass
