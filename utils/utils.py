import time

def calculate_speed(car_object, tutor_length):
    end_time = time.time()
    start_time = car_object['start_time']
    elapsed_time = end_time - start_time
    kmh_speed = tutor_length / elapsed_time * 3.6
    car_object['speed'] = kmh_speed