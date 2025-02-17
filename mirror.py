import numpy as np
import matplotlib.pyplot as plt

def triangle_area(p1, p2, p3):
    """
    Вычисляет площадь треугольника по трём 3D точкам.
    
    :param p1, p2, p3: np.array([x, y, z]) - координаты вершин треугольника
    :return: float - площадь треугольника
    """
    # Вычисляем векторы двух сторон треугольника
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    
    # Векторное произведение даёт параллелограмм, площадь которого равна длине вектора
    cross_product = np.cross(v1, v2)
    
    # Площадь треугольника - половина длины этого вектора
    area = np.linalg.norm(cross_product) / 2.0
    return area

def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def find_normal(object, intersection, ray_origin):
    if nearest_object['type'] =='sphere':
        normal = normalize(intersection - object['center'])
        if np.linalg.norm(object['center'] - ray_origin) < object['radius']:
            return -normal
        return normal
    if object['type'] == 'rectangle': 
        normal = normalize(np.cross(object['left_top'] - object['left_down'], object['right_top'] - object['left_top']))
        if np.linalg.norm(intersection + normal) > np.linalg.norm(intersection - normal):
            normal = - normal
        if object['is_glass']:
            normal = - normal
        return normal
    if nearest_object['type'] == 'linz':
        ... #hardcore

def rectangle_interest(ray_origin, ray_dir, rect_coords):
    """
    Находит расстояние от начала луча до ближайшей точки пересечения с прямоугольником.
    
    :param ray_origin: Начало луча (np.array([x, y, z]))
    :param ray_dir: Направление луча (np.array([x, y, z]))
    :param rect_coords: Четыре точки, задающие прямоугольник (список из 4 np.array)
    :return: Расстояние до точки пересечения или None, если пересечения нет.
    """
    # Нормализуем направление луча
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    
    # Берём три точки из четырёх
    p0, p1, p2, p3 = rect_coords
    
    # Векторные направления прямоугольника
    edge1 = p1 - p0
    edge2 = p3 - p0
    
    # Находим нормаль к плоскости
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)
    
    # Вычисляем знаменатель для уравнения пересечения
    denom = np.dot(normal, ray_dir)
    if np.abs(denom) < 1e-6:
        return None  # Луч параллелен плоскости
    
    # Вычисляем расстояние до плоскости
    d = np.dot(normal, p0 - ray_origin) / denom
    if d < 0:
        return None  # Пересечение позади начала луча
    
    # Точка пересечения с плоскостью
    intersection = ray_origin + d * ray_dir
    
    if triangle_area(p0, p1, intersection)+triangle_area(p2, p3, intersection)+triangle_area(p0, p2, intersection)+triangle_area(p1, p3, intersection) -(triangle_area(p0, p1, p2)+triangle_area(p2, p3, p1))< 1e-6:
        return d
    return None

def sphere_intersect(glassed ,center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            if glassed:
                return max(t1, t2)
            return min(t1, t2)
    return None


def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = []
    for obj in objects:
        if obj['type'] == 'sphere':
            distances.append(sphere_intersect(False,obj['center'], obj['radius'], ray_origin, ray_direction))
        if obj['type'] == 'rectangle':
            distances.append(rectangle_interest(ray_origin, ray_direction,(obj['left_down'], obj['left_top'], obj['right_down'], obj['right_top'])))
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance

width = 600
height = 300
max_depth = 5

camera = np.array([0, 0, 1])
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio) # left, top, right, bottom

light = { 'position': np.array([0, 0, 3]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }

objects = [
    { 'is_glass': False, 'type': 'sphere', 'center': np.array([-0.2, 0, -0.5]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 0, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 1 },
    #{ 'is_glass': False, 'type': 'sphere', 'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'is_glass': True, 'type': 'sphere', 'center': np.array([0, 0, 0.5]), 'radius': 0.15, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.2, 0.2, 0.2]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.3 },
    #{ 'is_glass': False, 'type': 'sphere', 'center': np.array([-35000, 0, -40000]), 'radius': 53149.0, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.8 },
    #{ 'type': 'sphere', 'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0 },
    #{ 'is_glass': False, 'type': 'rectangle', 'left_top': np.array([0.3, 0.2, -1.5]), 'right_top': np.array([0.6, 0.2, -0.2]), 'left_down': np.array([0.3, 0, -1.5]), 'right_down': np.array([0.6, 0, -0.2]), 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 10, 'reflection': 0.8 },
    #{ 'type': 'sphere', 'center': np.array([3000, 0, -4000]), 'radius': 4998.5, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.8 }
]

image = np.zeros((height, width, 3))
for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):

        # экран в начальной точке

        pixel = np.array([x, y, 0])
        origin = camera
        direction = normalize(pixel - origin)

        color = np.zeros((3))
        reflection = 1

        for k in range(max_depth):

            # проверка пересечений

            nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
            if nearest_object is None:

                break


            intersection = origin + min_distance * direction
            normal_to_surface = find_normal(nearest_object, intersection, origin) 
            shifted_point = intersection + 1e-5 * normal_to_surface
            intersection_to_light = normalize(light['position'] - shifted_point)

            _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)

            intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
            is_shadowed = min_distance < intersection_to_light_distance

            if is_shadowed:
                ... #break

            illumination = np.zeros((3))


            # ambiant

            illumination += nearest_object['ambient'] * light['ambient']

            # diffuse

            illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface)

            # specular

            intersection_to_camera = normalize(camera - intersection)
            H = normalize(intersection_to_light + intersection_to_camera)
            illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4)

            # reflection

            color += reflection * illumination
            reflection *= nearest_object['reflection']
            
            if nearest_object['is_glass']:
                dist = sphere_intersect(True,nearest_object['center'], nearest_object['radius'], origin, direction)
                intersection = origin + dist * direction
                normal_to_surface = find_normal(nearest_object, intersection, origin) 
                shifted_point = intersection + 1e-5 * normal_to_surface
                origin = shifted_point
                direction = direction  #+ 0.5*normal_to_surface
            else: 
                origin = shifted_point
                direction = reflected(direction, normal_to_surface)
        image[i, j] = np.clip(color, 0, 1)
    print("%d/%d" % (i + 1, height))
plt.imsave('image.png', image)
