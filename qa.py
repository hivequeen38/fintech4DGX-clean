import pandas as pd
from datetime import datetime
import trendMainDeltafromToday

# print('> Start QA ')

# symbol = 'TEST'
# currentDateTime = datetime.now()
# dateStr = currentDateTime.strftime('%Y-%m-%d %H:%M')
# closing_price = 115.30
# incr_array = [dateStr, closing_price, 'UP', 'DN', 'UP', 'DN', '__', '__', 'UP', '__', 'DN', 'UP', '__', 'DN', 'UP', '__', 'DN']
# input_col = ['date', 'close', 'p1', 'p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15']
# incr_df = pd.DataFrame([incr_array], columns=input_col)
# incr_df = incr_df.reset_index(drop=True)

# trendMainDeltafromToday.processDeltaFromTodayResults( symbol, incr_df, dateStr, closing_price)


def handle_flexible_master_sentiment(new_url: str):

    # Step 1: Read the CSV file into a DataFrame
    file_path = 'master_urls.csv'  # Replace with your actual file path
    df = pd.read_csv(file_path, header=None, names=['URL'])

    # Step 2: Wrap each URL in an <a> tag to make it clickable
    df['URL'] = df['URL'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')

    # Step 3: Append a new URL to the top
    df = pd.concat([pd.DataFrame({'URL': [f'<a href="{new_url}" target="_blank">{new_url}</a>' ]}), df], ignore_index=True)

    # Step 4: Keep only the first 100 entries
    df = df.head(100)

    # Step 5: Generate a HTML table with a larger font size using Styler
    styled_df = df.style.set_table_styles([
        {'selector': 'table', 'props': [('font-size', '16px')]}
    ])

    # Use to_html() to render the styled DataFrame to HTML with escape=False to prevent escaping of HTML tags
    html_table = styled_df.to_html(escape=False)

    # Step 6: (Optional) Save the HTML table to a file
    with open('urls_table.html', 'w') as file:
        file.write(html_table)

    # Display the generated HTML table (optional)
    print(html_table)


# Main flow
new_utl = 'www.test1.html'
handle_flexible_master_sentiment(new_utl)