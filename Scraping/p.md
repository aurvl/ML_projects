# Movie Information Scraping from AlloCine

## Project Overview
This project aims to extract movie information from AlloCine by scraping HTML pages stored locally. The collected data is stored in a structured format (Pandas DataFrame), which will then be used for further analysis and modeling. The first phase focuses on retrieving movie details such as title, duration, genre, director, ratings, and more from each page.

## Goals
1. **Scraping Movie Data**: Extract data from 57 movie pages stored locally.
2. **Data Storage**: Store the scraped data in a DataFrame for further analysis.
3. **Future Analysis**:
   - Visualizing the scraped data using Power BI.
   - Building a machine learning model to predict one of the scraped variables (e.g., audience rating, number of seasons).

## Data Fields
The information extracted for each movie includes:
- **Title**: Movie title
- **Status**: Whether the movie is completed or ongoing
- **Period**: Release and end dates
- **Duration**: Movie duration (in minutes)
- **Type/Genre**: Genre of the movie
- **Director**: Name(s) of the director
- **Main Character(s)**: Leading actor in the movie
- **Nationality**: Country of origin
- **Channel**: The original broadcasting channel
- **Press Rating**: Ratings provided by the press
- **Audience Rating**: Ratings provided by the audience
- **Number of Seasons and Episodes**: For series, the total number of seasons and episodes
- **Description**: Short description or synopsis of the movie

## Step-by-Step Process

### Step 1: Scraping and Storing Data
Using regular expressions, the following data fields were extracted from the HTML pages:
- Titles, genres, ratings, director names, etc.

These fields were then organized into a Pandas DataFrame. Below is a sample row from the resulting DataFrame:

| Title    | Status | Period   | Duration | Genre   | Director     | Audience Rating | Seasons | Episodes | Description       |
|----------|--------|----------|----------|---------|--------------|-----------------|---------|----------|-------------------|
| El Barco | None   | 2011-2013| 75 min   | Aventure| Iván Escobar | 3.8             | None    | None     | None              |

### Step 2: Future Work
The next steps in this project will involve:
1. **Power BI Report**: Visualizing the data by creating an interactive dashboard showcasing key statistics and trends across the movies.
2. **Machine Learning Model**: Using the dataset to build a machine learning model to predict one of the variables, such as audience rating or number of seasons.

## Tools and Libraries
- **Python**
  - `re` for regular expression-based scraping
  - `pandas` for data manipulation
  - `html` for handling special characters in HTML files
  - `os` for managing file directories
- **Power BI** (for future analysis)
- **Machine Learning Framework** (to be determined in the next step)

## How to Run
1. Place the HTML files in the `Data/Pages` directory.
2. Run the scraping script to extract movie information into a DataFrame.
3. Export the DataFrame to a CSV file for further use in Power BI or modeling.

```bash
python scrape_allocine.py
```

## Future Plans
- **Power BI Dashboard**: Once the data is collected, it will be used to build a dynamic report in Power BI.
- **Machine Learning**: Explore different models to predict the audience rating or another relevant variable from the dataset.

## References
- Data sourced from [AlloCine](https://www.allocine.fr)
