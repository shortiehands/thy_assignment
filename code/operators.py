import copy
import random

import numpy as np
from psp import PSP


### Destroy operators ###
# You can follow the example and implement destroy_2, destroy_3, etc
def destroy_1(current: PSP, random_state):
    """Destroy operator sample (name of the function is free to change)
    Args:
        current::PSP
            a PSP object before destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        destroyed::PSP
            the PSP object after destroying
    """
    destroyed = current.copy()
    
    removal_frac = 0.15

    # Gather all tasks currently assigned to workers
    assigned_tasks = []
    for w in destroyed.workers:
        assigned_tasks += w.tasks_assigned

    # Determine how many tasks to remove
    num_remove = int(len(assigned_tasks) * removal_frac)
    if num_remove < 1:
        return destroyed  # If there's <1 to remove, just return unchanged

    # Randomly select tasks to remove
    tasks_to_remove = random_state.choice(assigned_tasks, num_remove, replace=False)

    # Remove them from the copy
    for task in tasks_to_remove:
        for w in destroyed.workers:
            if task in w.tasks_assigned:
                w.remove_task(task.id)
                break

    # Update destroyed.unassigned accordingly
    assigned_now = [t for w in destroyed.workers for t in w.tasks_assigned]
    destroyed.unassigned = [t for t in destroyed.tasks if t not in assigned_now]

    return destroyed
def destroy_by_most_expensive(current: PSP, random_state):
    """Destroy operator that prioritizes removing expensive tasks
    Args:
        current::PSP
            a PSP object before destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        destroyed::PSP
            the PSP object after destroying
    """
    destroyed = current.copy()
    
    removal_frac = 0.2

    # Calculate cost contribution for each task
    task_costs = []
    for w in destroyed.workers:
        for task in w.tasks_assigned:
            # Store the original cost
            original_cost = w.get_objective()
            
            # Temporarily remove the task to see how cost changes
            w.remove_task(task.id)
            new_cost = w.get_objective()
            cost_saving = original_cost - new_cost
            
            # Put the task back
            w.assign_task(task)
            
            # Store the task and its cost contribution
            task_costs.append((task, cost_saving))
    
    # Sort tasks by cost contribution (most expensive first)
    task_costs.sort(key=lambda x: x[1], reverse=True)
    
    # Determine how many tasks to remove
    num_remove = int(len(task_costs) * removal_frac)
    if num_remove < 1:
        return destroyed  # If there's <1 to remove, just return unchanged
    
    # Take the most expensive tasks
    tasks_to_remove = [task for task, _ in task_costs[:num_remove]]
    
    # Remove them from the copy
    for task in tasks_to_remove:
        for w in destroyed.workers:
            if task in w.tasks_assigned:
                w.remove_task(task.id)
                break

    # Update destroyed.unassigned accordingly
    assigned_now = [t for w in destroyed.workers for t in w.tasks_assigned]
    destroyed.unassigned = [t for t in destroyed.tasks if t not in assigned_now]

    return destroyed


### Repair operators ###
# You can follow the example and implement repair_2, repair_3, etc
def repair_1(destroyed: PSP, random_state):
    """repair operator sample (name of the function is free to change)
    Args:
        destroyed::PSP
            a PSP object after destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        repaired::PSP
            the PSP object after repairing
    """
    """
    Greedily reassign unassigned tasks in ascending day/hour order.
    """
    new_state = destroyed.copy()
    
    # sort unassigned by (day, hour)
    new_state.unassigned.sort(key=lambda t: (t.day, t.hour))
    still_unassigned = []
    
    for task in new_state.unassigned:
        best_worker = None
        best_increase = float('inf')
        
        for w in new_state.workers:
            if w.can_assign(task):
                # compute cost increase if assigned
                old_obj = w.get_objective()
                
                # temp assign
                w.assign_task(task)
                new_obj = w.get_objective()
                
                # revert
                w.remove_task(task.id)
                
                cost_increase = new_obj - old_obj
                if cost_increase < best_increase:
                    best_increase = cost_increase
                    best_worker = w
        
        if best_worker:
            best_worker.assign_task(task)
        else:
            still_unassigned.append(task)
    
    # update unassigned
    new_state.unassigned = still_unassigned
    return new_state

def repair_best_improvement(destroyed: PSP, random_state):
    new_state = destroyed.copy()
    
    # Helper function to determine task priority
    def task_priority(task):
        eligible_workers = []
        for w in new_state.workers:
            if w.can_assign(task):
                eligible_workers.append(w)
        
        # Count of eligible workers (fewer is higher priority)
        count = len(eligible_workers)
        
        # Average rate of eligible workers (lower is better)
        avg_rate = sum(w.rate for w in eligible_workers) / count if count > 0 else float('inf')
        
        return (count, avg_rate, task.day, task.hour)
    
    # Sort unassigned tasks by: fewer eligible workers, lower average cost, then day/hour
    new_state.unassigned.sort(key=task_priority)
    still_unassigned = []
    
    for task in new_state.unassigned:
        best_worker = None
        best_increase = float('inf')
        
        for w in new_state.workers:
            if w.can_assign(task):
                original_obj = w.get_objective()
                w.assign_task(task)
                new_obj = w.get_objective()
                cost_increase = new_obj - original_obj
                # Remove task again to test other possibilities
                w.remove_task(task.id)
                
                if cost_increase < best_increase:
                    best_increase = cost_increase
                    best_worker = w
        
        if best_worker:
            best_worker.assign_task(task)
        else:
            still_unassigned.append(task)
    
    new_state.unassigned = still_unassigned
    return new_state

