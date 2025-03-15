import os
import regex as re
from psp import PSP, Worker, Task, Parser
import argparse


def parse_output(output):
    # Add parsing logic here
    
    # Extract worker-task assignments
    workers = {}  # worker: ([tasks], {day: (start, end)})
    pattern = r'Worker (\d+): Day (\d+) Hours \[(.*?)\] Tasks \[(.*?)\]'
    p = re.compile(pattern)
    matches = p.findall(output)
    
    for match in matches:
        # get input values
        worker = int(match[0])
        day = int(match[1])
        hours = [int(h) for h in match[2].split(', ') if h]
        tasks = [int(t) for t in match[3].split(', ') if t]
        
        # add worker to dict if not existing
        if worker not in workers:
            workers[worker] = ([], {})  # (tasks, {day: (start, end)})
        
        # add tasks to worker
        workers[worker][0].extend(tasks)
        
        # add day to worker with hours block
        if day not in workers[worker][1].keys():
            workers[worker][1][day] = (hours[0], hours[1])
        else:
            curr_hours = workers[worker][1][day]
            workers[worker][1][day] = (min(curr_hours[0], hours[0]), max(curr_hours[1], hours[1]))
    print('Successfully parsed output.')
    return workers


def validate_output(solution, input):
    errors = {}  # worker: error messages
    # Add validation logic here
    # for each worker, validate skillset, availability, multitasking, rest_time, block limits and total time limits
    for worker_id, values in solution.items():
        worker = input.workers[worker_id]
        tasks = values[0]
        times = values[1]
        
        worker_errors = []
        
        # for each task of the worker, check skillset and availability
        for task_id in tasks:
            task = input.tasks[task_id]
            
            # checking skillset
            if task.skill not in worker.skills:
                worker_errors.append(f'Wrong skillset for Task {task_id}.')
            
            # checking availability
            avail_hours = worker.available.get(task.day)
            if not avail_hours:
                worker_errors.append(f'Not Available on this day for Task {task_id}.')
            #   ## check if task.hour within possible hours for current day
            if task.hour < avail_hours[0] or task.hour > avail_hours[1]:
                worker_errors.append(f'Not Available on this hpur for Task {task_id}.')
            
        # checking for multitasking
        if len(tasks) > 1:
            # check if tasks overlap
            for i in range(len(tasks)):
                for j in range(i+1, len(tasks)):
                    task1 = input.tasks[tasks[i]]
                    task2 = input.tasks[tasks[j]]
                    if task1.day == task2.day and task1.hour == task2.hour:
                        worker_errors.append(f'Tasks {tasks[i]} and {tasks[j]} overlap.')
        
        # checking rest_time and time limits for each block and in total
        total_hours = 0
        for day, hours in times.items():
            # check rest time
            prev_day, next_day = day - 1, day + 1
            prev_hours, next_hours = times.get(prev_day), times.get(next_day)
            if prev_hours:
                if 23 - prev_hours[1] + hours[0] < worker.rmin:
                    worker_errors.append(f'Worker does not meet rest time on Day {day}.')
            if next_hours:
                if 23 - hours[1] + next_hours[0] < worker.rmin:
                    worker_errors.append(f'Worker does not meet rest time on Day {day}.')
            # check block limit
            if hours[1] - hours[0] + 1 > worker.bmax:
                worker_errors.append(f'Worker exceeds block limit on Day {day}.')
            total_hours += hours[1] - hours[0] + 1
        # check total time limit
        if total_hours > worker.wmax:
            worker_errors.append(f'Worker exceeds total time limit with {total_hours} over the limit of {worker.wmax}.')
        
        # if there are errors, add to dict
        if worker_errors:
            errors[worker_id] = worker_errors
    
    # check if tasks are assigned more than once
    task_errors = []
    assigned_tasks = [task_id for values in solution.values() for task_id in values[0]]
    seen = set()
    for t in assigned_tasks:
        if t in seen:
            task_errors.append(f'Task {t} is assigned more than once.')
        seen.add(t)
    if task_errors:
        errors['tasks'] = task_errors
            
    print('Successfully validated output.')
    return errors


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='load data')
    parser.add_argument(dest='solution', type=str, help='solution file')
    parser.add_argument(dest='input', type=str, help='input instance file')
    args = parser.parse_args()
    
    solution_directory = '/Users/panda/Desktop/MITB/10_AIPlanning/Assignment/code/'
    input_directory = '/Users/panda/Desktop/MITB/10_AIPlanning/Assignment/code/psp_instances/sample_instances'
    
    solution = os.path.join(solution_directory, args.solution)
    if os.path.isfile(solution):
        with open(solution, 'r') as file:
            output = file.read()
        sol_workers = parse_output(output)
        # print data to validate
        # for worker, values in sol_workers.items():
        #     print(f'Worker {worker}:')
        #     print(f'Tasks: {values[0]}')
        #     print(f'Days: {values[1]}')
        print('Successfully loaded solution instance.')
    
    input = os.path.join(input_directory, args.input)
    if os.path.isfile(input):
        # load data
        parsed = Parser(input)
        psp = PSP(parsed.name, parsed.workers, parsed.tasks, parsed.Alpha)
        print('Successfully loaded input instance.')
    
    # validate output
    if sol_workers and psp:
        errors = validate_output(sol_workers, psp)
        if errors:
            print('Output is invalid.')
            
            for worker, error_msgs in errors.items():
                print(f'Worker {worker}:')
                for msg in error_msgs:
                    print(f'  - {msg}')

        else:
            print('Output is valid.')
    else:
        print('Error loading input instance or parsing output.')
