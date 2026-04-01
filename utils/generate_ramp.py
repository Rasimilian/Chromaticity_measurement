import numpy as np


def generate_base_ramp(e_inj, e_ext, e_null=0, length=9838):
    """
    Генерирует массив для скана

    Args:
        e_inj: значение для первых 500 точек и начало/конец
        e_ext: значение в пике синуса и на полочке
        e_null: промежуточное значение (по умолчанию 0)
        length: общая длина массива (по умолчанию 9838)

    Returns:
        numpy.array: сгенерированный массив
    """
    result = np.zeros(length)

    n_points_sine = 3500  # длина первого и второго синуса
    n_points_sine3 = 600  # длина третьего синуса

    # 1. Первые 500 точек = e_inj
    result[:500] = e_inj

    # 2. Первый синус: подъем от e_inj до e_ext (от -π/2 до π/2)
    x1 = np.linspace(-np.pi/2, np.pi/2, n_points_sine)
    sine_norm1 = (np.sin(x1) + 1) / 2  # от 0 до 1
    sine_part1 = e_inj + (e_ext - e_inj) * sine_norm1
    result[500:500+n_points_sine] = sine_part1

    # Проверка стыка: result[500] = e_inj, result[500+n_points_sine-1] = e_ext

    # 3. Полочка e_ext (500 точек)
    result[4000:4500] = e_ext  # 4000 = 500 + 3500, 4500 = 4000 + 500

    # Проверка стыка: result[4000] = e_ext (конец первого синуса), result[4499] = e_ext

    # 4. Второй синус: спад от e_ext до e_null (от π/2 до -π/2)
    x2 = np.linspace(np.pi/2, -np.pi/2, n_points_sine)
    sine_norm2 = (np.sin(x2) + 1) / 2  # от 1 до 0
    sine_part2 = e_null + (e_ext - e_null) * sine_norm2
    result[4500:4500+n_points_sine] = sine_part2  # начинаем с 4500, где было e_ext

    # Проверка стыка: result[4500] = e_ext, result[4500+n_points_sine-1] = e_null

    # 5. Полочка e_null (400 точек)
    result[8000:8400] = e_null  # 8000 = 4500 + 3500, 8400 = 8000 + 400

    # Проверка стыка: result[8000] = e_null (конец второго синуса)

    # 6. Третий синус: подъем от e_null до e_inj за 600 точек (от -π/2 до π/2)
    x3 = np.linspace(-np.pi/2, np.pi/2, n_points_sine3)
    sine_norm3 = (np.sin(x3) + 1) / 2  # от 0 до 1
    sine_part3 = e_null + (e_inj - e_null) * sine_norm3
    result[8400:8400+n_points_sine3] = sine_part3  # начинаем с 8400, где было e_null

    # Проверка стыка: result[8400] = e_null, result[8400+n_points_sine3-1] = e_inj

    # 7. Полочка e_inj до конца массива
    result[8400+n_points_sine3:] = e_inj  # с 9000 = 8400 + 600

    return np.array(result) / e_inj