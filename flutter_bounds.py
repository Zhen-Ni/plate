#!/usr/bin/env python3

import plate


def find_critical_lambda(p: plate.SupersonicPlate,
                         lambda_lower=10, lambda_upper=1000,
                         order=2,
                         tol=0.1):
    while not is_stable(p, lambda_lower, order):
        lambda_lower /= 2
    while is_stable(p, lambda_upper, order):
        lambda_upper *= 2
    lambda_middle = (lambda_upper + lambda_lower) / 2
    while tol < lambda_upper - lambda_lower:
        if is_stable(p, lambda_middle, order):
            lambda_lower = lambda_middle
        else:
            lambda_upper = lambda_middle
        lambda_middle = (lambda_upper + lambda_lower) / 2
    return lambda_middle
    

def is_stable(p: plate.SupersonicPlate, lambda_, order):
    p.reset_airflow()
    p.add_airflow_lambda(lambda_)
    solver = plate.ComplexModalSolver(p)
    res = solver.solve(order)
    return res.stable.all()
