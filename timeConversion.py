from datetime import datetime
import pytz

# Example timestamp
timestamp_str = "20240129T175100"  # Format: yyyymmddThhmmss

# Convert the string to a datetime object
timestamp = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")

# Set the timezone to UTC
timestamp_utc = pytz.utc.localize(timestamp)

# Convert to US Pacific Time
pacific = pytz.timezone("US/Pacific")
timestamp_pacific = timestamp_utc.astimezone(pacific)

# Format the timestamp as a string in 'YYYYMMDDTHHMM' format
formatted_timestamp = timestamp_pacific.strftime("%Y%m%dT%H%M")

print(formatted_timestamp)
