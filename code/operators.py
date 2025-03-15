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
    
    removal_frac = 0.2

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

    # 7. Update destroyed.unassigned accordingly
    assigned_now = [t for w in destroyed.workers for t in w.tasks_assigned]
    destroyed.unassigned = [t for t in destroyed.tasks if t not in assigned_now]

    return destroyed

def destroy_block_based(current: PSP, random_state):
    destroyed = current.copy()
    # For each worker with a block assigned, randomly select a day and remove all tasks on that day.
    for w in destroyed.workers:
        if w.blocks:
            day = random_state.choice(list(w.blocks.keys()))
            # Remove every task assigned on that day.
            tasks_in_day = [t for t in w.tasks_assigned if t.day == day]
            for task in tasks_in_day:
                w.remove_task(task.id)
    # Update unassigned tasks list.
    assigned_now = [t for w in destroyed.workers for t in w.tasks_assigned]
    destroyed.unassigned = [t for t in destroyed.tasks if t not in assigned_now]
    return destroyed

def destroy_cost_based(current: PSP, random_state):
    destroyed = current.copy()
    removal_frac = 0.2  # Remove 20% of assignments
    task_marginal_cost = []
    
    # Evaluate each assigned task for its marginal cost impact
    for w in destroyed.workers:
        for task in w.tasks_assigned:
            original_cost = w.get_objective()
            # Temporarily remove the task to compute the change
            w.remove_task(task.id)
            new_cost = w.get_objective()
            cost_diff = original_cost - new_cost  # improvement if removed
            # Reassign task for now so we can decide later
            w.assign_task(task)
            task_marginal_cost.append((task, cost_diff))
    
    # Sort tasks: highest cost contribution first
    task_marginal_cost.sort(key=lambda x: x[1], reverse=True)
    num_remove = int(len(task_marginal_cost) * removal_frac)
    tasks_to_remove = [t for t, _ in task_marginal_cost[:num_remove]]
    
    # Remove the selected tasks
    for task in tasks_to_remove:
        for w in destroyed.workers:
            if task in w.tasks_assigned:
                w.remove_task(task.id)
                break
    
    # Update the list of unassigned tasks
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
    # Process unassigned tasks in order (e.g., by day and hour)
    new_state.unassigned.sort(key=lambda t: (t.day, t.hour))
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
                # Remove task again to test other possibilities.
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

def swap_reassign_operator(current: PSP, random_state):
    new_state = current.copy()
    # Randomly pick two workers.
    if len(new_state.workers) < 2:
        return new_state
    w1, w2 = random_state.choice(new_state.workers, 2, replace=False)
    if not w1.tasks_assigned or not w2.tasks_assigned:
        return new_state
    # Randomly choose a task from each.
    task1 = random_state.choice(w1.tasks_assigned)
    task2 = random_state.choice(w2.tasks_assigned)
    
    # Check feasibility for swapping: worker1 can take task2 and worker2 can take task1.
    if w1.can_assign(task2) and w2.can_assign(task1):
        # Remove tasks from their original workers.
        w1.remove_task(task1.id)
        w2.remove_task(task2.id)
        # Attempt the swap.
        if w1.can_assign(task2) and w2.can_assign(task1):
            w1.assign_task(task2)
            w2.assign_task(task1)
        else:
            # If swap fails, revert.
            w1.assign_task(task1)
            w2.assign_task(task2)
    
    # Update unassigned tasks
    assigned_now = [t for w in new_state.workers for t in w.tasks_assigned]
    new_state.unassigned = [t for t in new_state.tasks if t not in assigned_now]
    return new_state


