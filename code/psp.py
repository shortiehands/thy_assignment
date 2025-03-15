import copy
import json
import random

from src.alns import State


### Parser to parse instance json file ###
# You should not change this class!
class Parser(object):
    def __init__(self, json_file):
        """initialize the parser, saves the data from the file into the following instance variables:
        -
        Args:
            json_file::str
                the path to the xml file
        """
        self.json_file = json_file
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.name = self.data["name"]
        self.Alpha = self.data["ALPHA"]
        self.T = self.data["T"]
        self.BMAX = self.data["BMax"]
        self.WMAX = self.data["WMax"]
        self.RMIN = self.data["RMin"]

        self.workers = [
            Worker(worker_data, self.T, self.BMAX, self.WMAX, self.RMIN)
            for worker_data in self.data["Workers"]
        ]
        self.tasks = [Task(task_data) for task_data in self.data["Tasks"]]


class Worker(object):
    def __init__(self, data, T, bmax, wmax, rmin):
        """Initialize the worker
        Attributes:
            id::int
                id of the worker
            skills::[skill]
                a list of skills of the worker
            available::{k: v}
                key is the day, value is the list of two elements,
                the first element in the value is the first available hour for that day,
                the second element in the value is the last available hour for that day, inclusively
            bmax::int
                maximum length constraint
            wmax::int
                maximum working hours
            rmin::int
                minimum rest time
            rate::int
                hourly rate
            tasks_assigned::[task]
                a list of task objects
            blocks::{k: v}
                key is the day where a block is assigned to this worker
                value is the list of two elements
                the first element is the hour of the start of the block
                the second element is the hour of the start of the block
                if a worker is not assigned any tasks for the day, the key is removed from the blocks dictionary:
                        Eg. del self.blocks[D]

            total_hours::int
                total working hours for the worker

        """
        self.id = data["w_id"]
        self.skills = data["skills"]
        self.T = T
        self.available = {int(k): v for k, v in data["available"].items()}
        # the constant number for f2 in the objective function
        self.bmin = 4
        self.bmax = bmax
        self.wmax = wmax
        self.rmin = rmin

        self.rate = data["rate"]
        self.tasks_assigned = []
        self.blocks = {}
        self.total_hours = 0

    def can_assign(self, task):
        ## 1. Check skill set
        if task.skill not in self.skills:
            return False

        ## 2. Check available time slots
        if task.day not in self.available:
            return False
        
        day_start, day_end = self.available[task.day]
        if not (day_start <= task.hour <= day_end):
            return False
        
        ## 3. Check if worker is already assigned a task at this time
        for assigned_task in self.tasks_assigned:
            if assigned_task.day == task.day and assigned_task.hour == task.hour:
                return False
        
        ## 4. Check if task has already been assigned to this worker
        if task.id in [t.id for t in self.tasks_assigned]:
            return False
        
        ## 5. Calculate potential new block boundaries if this task is assigned
        if task.day not in self.blocks:
            new_start = task.hour
            new_end = task.hour
        else:
            current_start, current_end = self.blocks[task.day]
            new_start = min(current_start, task.hour)
            new_end = max(current_end, task.hour)
        
        ## 6. Check block length constraint
        block_length = new_end - new_start + 1
        if block_length > self.bmax:
            return False
        
        ## 7. Check total working hours constraint
        if task.day not in self.blocks:
            additional_hours = block_length
        else:
            current_start, current_end = self.blocks[task.day]
            current_length = current_end - current_start + 1
            additional_hours = block_length - current_length
        
        if self.total_hours + additional_hours > self.wmax:
            return False
        
        ## 8. Check rest constraint with previous day's block
        for d in range(task.day - 1, -1, -1):
            if d in self.blocks:
                _, prev_end = self.blocks[d]
                prev_avail_end = self.available[d][1]
                curr_avail_start = self.available[task.day][0]
                # Calculate rest hours relative to available window
                rest_hours = (prev_avail_end - prev_end) + (new_start - curr_avail_start) + (task.day - d - 1) * 24
                if rest_hours < self.rmin:
                    return False
                break
        
        ## 9. Check rest constraint with next day's block
        for d in range(task.day+1, self.T):
            if d in self.blocks:
                next_start, _ = self.blocks[d]
                # Calculate rest time between new_end on task.day and next_start on day d
                # Rest = hours remaining in task.day + full days between + hours before next_start
                rest_hours = (24 - new_end) + (d - task.day - 1) * 24 + next_start
                if rest_hours < self.rmin:
                    return False
                break  # Only check the closest next day with a block
        
        return True

    def assign_task(self, task):
        # // Implement Code Here
        self.tasks_assigned.append(task)
        day = task.day
        hour = task.hour

        if day not in self.blocks:
            # New block: [hour, hour]
            self.blocks[day] = [hour, hour]
            self.total_hours += 1
        else:
            start, end = self.blocks[day]
            new_start = min(start, hour)
            new_end   = max(end, hour)
            # remove old block length
            old_len = (end - start + 1)
            self.total_hours -= old_len
            # add new block length
            new_len = (new_end - new_start + 1)
            self.total_hours += new_len
            # update block
            self.blocks[day] = [new_start, new_end]

    def remove_task(self, task_id):
        # Find the task in tasks_assigned
        removed_task = None
        for t in self.tasks_assigned:
            if t.id == task_id:
                removed_task = t
                break
        
        if not removed_task:
            return False  # Task not found
        
        # Remove task from assigned tasks
        self.tasks_assigned.remove(removed_task)
        day = removed_task.day
        
        # Recompute block for that day from the remaining tasks
        day_tasks = [t.hour for t in self.tasks_assigned if t.day == day]
        
        if not day_tasks:
            # No tasks left on this day, remove block
            if day in self.blocks:  # This check is safer
                old_block_len = (self.blocks[day][1] - self.blocks[day][0] + 1)
                self.total_hours -= old_block_len
                del self.blocks[day]
        else:
            # Adjust block to cover remaining tasks
            new_start = min(day_tasks)
            new_end = max(day_tasks)
            
            # Update total hours
            old_len = (self.blocks[day][1] - self.blocks[day][0] + 1)
            new_len = (new_end - new_start + 1)
            self.total_hours = self.total_hours - old_len + new_len
            
            # Update block
            self.blocks[day] = [new_start, new_end]
        
        return True
    def get_objective(self):
        t = sum(x[1] - x[0] + 1 for x in self.blocks.values())
        return t * self.rate

    def __repr__(self):
        if len(self.blocks) == 0:
            return ""
        return "\n".join(
            [
                f"Worker {self.id}: Day {d} Hours {self.blocks[d]} Tasks {sorted([t.id for t in self.tasks_assigned if t.day == d])}"
                for d in sorted(self.blocks.keys())
            ]
        )


class Task(object):
    def __init__(self, data):
        self.id = data["t_id"]
        self.skill = data["skill"]
        self.day = data["day"]
        self.hour = data["hour"]


### PSP state class ###
# PSP state class. You could and should add your own helper functions to the class
# But please keep the rest untouched!
class PSP(State):
    def __init__(self, name, workers, tasks, alpha):
        """Initialize the PSP state
        Args:
            name::str
                name of the instance
            workers::[Worker]
                workers of the instance
            tasks::[Task]
                tasks of the instance
        """
        self.name = name
        self.workers = workers
        self.tasks = tasks
        self.Alpha = alpha
        # the tasks assigned to each worker, eg. [worker1.tasks_assigned, worker2.tasks_assigned, ..., workerN.tasks_assigned]
        self.solution = []
        self.unassigned = list(tasks)

    def random_initialize(self, seed=None):
        """
        Improved construction heuristic: Tasks are processed in an order that 
        prioritizes those that are harder to assign (i.e. fewer eligible workers). 
        After an initial assignment pass, a multi-pass loop iterates over remaining 
        unassigned tasks until no more tasks can be assigned.
        """
        if seed is None:
            seed = 606
        random.seed(seed)

        # Helper: count number of workers eligible (based on skill and basic available time)
        def eligible_count(task):
            count = 0
            for w in self.workers:
                if task.skill in w.skills and task.day in w.available:
                    day_start, day_end = w.available[task.day]
                    if day_start <= task.hour <= day_end:
                        count += 1
            return count

        # Helper: for a given task, choose among feasible workers the one with the earliest new block end.
        def assign_task_to_best_worker(task):
            feasible_workers = []
            for w in self.workers:
                if w.can_assign(task):
                    # Calculate what the block's new end time would be if task is assigned.
                    if task.day in w.blocks:
                        new_end = max(w.blocks[task.day][1], task.hour)
                    else:
                        new_end = task.hour
                    feasible_workers.append((w, new_end))
            if feasible_workers:
                best_worker, _ = min(feasible_workers, key=lambda x: x[1])
                best_worker.assign_task(task)
                return True
            return False

        # First pass: sort tasks by (eligible_count, day, hour) so that tasks with fewer eligible workers are handled first.
        tasks_sorted = sorted(self.tasks, key=lambda t: (eligible_count(t), t.day, t.hour))
        for task in tasks_sorted:
            assign_task_to_best_worker(task)

        # Update unassigned tasks list.
        assigned_tasks = []
        for w in self.workers:
            assigned_tasks += w.tasks_assigned
        self.unassigned = [t for t in self.tasks if t not in assigned_tasks]

        # Multi-pass: repeatedly try to assign remaining unassigned tasks.
        improved = True
        while improved and self.unassigned:
            improved = False
            # Sort unassigned tasks by eligible_count (lowest count first)
            tasks_to_try = sorted(self.unassigned, key=lambda t: eligible_count(t))
            for task in tasks_to_try:
                if assign_task_to_best_worker(task):
                    improved = True
            # Recompute unassigned list.
            assigned_tasks = []
            for w in self.workers:
                assigned_tasks += w.tasks_assigned
            new_unassigned = [t for t in self.tasks if t not in assigned_tasks]
            # If we weren't able to assign any additional tasks in this pass, exit.
            if len(new_unassigned) == len(self.unassigned):
                break
            self.unassigned = new_unassigned



    def copy(self):
        return copy.deepcopy(self)

    def objective(self):
        """Calculate the objective value of the state
        Return the total cost of each worker + unassigned cost
        """
        f1 = len(self.unassigned)
        f2 = sum(max(worker.get_objective(), 50) for worker in self.workers if worker.get_objective() > 0)
        return self.Alpha * f1 + f2
