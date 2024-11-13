import random

# Problem parameters
num_customers = 10
num_vehicles = 3
vehicle_capacity = 15
customer_demands = [random.randint(1, 5) for _ in range(num_customers)]
distance_matrix = [[random.uniform(1, 10) for _ in range(num_customers + 1)] for _ in range(num_customers + 1)]

# Particle class representing a potential solution
class Particle:
    def __init__(self):
        self.position = self.random_solution()
        self.velocity = []
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

    def random_solution(self):
        # Randomly assign customers to vehicles
        customers = list(range(1, num_customers + 1))  # Assuming depot is 0
        random.shuffle(customers)
        routes = [[] for _ in range(num_vehicles)]
        for customer in customers:
            routes[random.randint(0, num_vehicles - 1)].append(customer)
        return routes

    def fitness(self):
        total_distance = 0
        for route in self.position:
            load = sum(customer_demands[customer - 1] for customer in route)
            if load > vehicle_capacity:
                return float('inf')  # Invalid solution
            distance = 0
            prev_location = 0  # Start from depot
            for customer in route:
                distance += distance_matrix[prev_location][customer]
                prev_location = customer
            distance += distance_matrix[prev_location][0]  # Return to depot
            total_distance += distance
        return total_distance

    def update_velocity(self, global_best_position):
        # Implement velocity update logic specific to VRP
        pass

    def update_position(self):
        # Implement position update logic specific to VRP
        pass

# PSO algorithm
swarm_size = 30
swarm = [Particle() for _ in range(swarm_size)]
global_best_position = None
global_best_fitness = float('inf')

for iteration in range(100):
    for particle in swarm:
        fitness = particle.fitness()
        if fitness < particle.best_fitness:
            particle.best_fitness = fitness
            particle.best_position = particle.position.copy()
        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best_position = particle.position.copy()
    for particle in swarm:
        particle.update_velocity(global_best_position)
        particle.update_position()

print("Best total distance:", global_best_fitness)
print("Best routes:")
for route in global_best_position:
    print("Route:", route)