"""logging.py: tools for generating logs.
"""

import datetime

# print date
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
current_date = datetime.datetime.now()
formatted_date = str(current_date.strftime('%Y-%m-%d %H:%M:%S')) + ", " + \
                 weekdays[int(current_date.isoweekday()) - 1]
print(formatted_date)

