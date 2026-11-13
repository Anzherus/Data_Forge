"""Кастомные исключения проекта DataForge."""

class DataForgeError(Exception):
    """Базовое исключение проекта."""


class ConfigError(DataForgeError):
    """Ошибка конфигурации."""


class StorageError(DataForgeError):
    """Ошибка хранилища (SQLite)."""


class LoadError(DataForgeError):
    """Ошибка загрузки файла."""


class UnsupportedFormatError(LoadError):
    """Неподдерживаемый формат файла."""


class EncodingError(LoadError):
    """Ошибка кодировки."""


class CorruptedFileError(LoadError):
    """Повреждённый файл (битый JSON, CSV и т.п.)."""


class SchemaError(DataForgeError):
    """Несовместимая структура данных."""


class ValidationError(DataForgeError):
    """Ошибка валидации данных."""


class TransformError(DataForgeError):
    """Ошибка преобразования."""


class PipelineError(DataForgeError):
    """Ошибка выполнения сценария."""


class ExportError(DataForgeError):
    """Ошибка экспорта."""


class BenchmarkError(DataForgeError):
    """Ошибка бенчмарка."""