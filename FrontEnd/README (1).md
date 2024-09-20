 

## Frontend Repository

### Screenshot of the Top Page with Favorites Stars Displayed
![image](https://github.com/user-attachments/assets/e4cfa8e1-5492-454d-ae5d-e3c85f49c60b)


### Screenshot of the Favorites List Database Table Displayed in MongoDB Compass
 
 TO BE PASTED

 
 
 
# Movie Review App - Part 4: Authentication


## Overview

In this part of the project, I set up authentication to enable users to log in and log out, ensuring that certain functionalities (specifically the ability to create, edit, and delete reviews) are available only when users are logged in. I used Google's authentication API to handle this. Anyone with a Gmail account will automatically be able to identify themselves to our application.
## Available Scripts


1. A screenshot of a movie reviews page with a newly written review:

   
<img width="1410" alt="Screenshot 2024-07-08 at 5 45 49 PM" src="https://media.github.khoury.northeastern.edu/user/17515/files/a91de406-3e39-445e-82d0-9d97fdf4ad1f">


2. A screenshot of the same page with the review edited:

 


# Movie App Assignment - Part 3 

## Description
This repository contains the frontend for a Movie App. The app allows users to browse a list of movies, view detailed information about each movie, and see reviews. This assignment focuses on demonstrating the integration of a mock server for testing purposes and ensuring that the frontend correctly renders data.


### 1. App at localhost:3000 with at least one movie showing the placeholder poster
The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

<img width="1431" alt="Screenshot 2024-06-29 at 12 51 04 AM" src="https://media.github.khoury.northeastern.edu/user/17515/files/48144aa4-9bc3-446a-bf16-37991cf852f9">
See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### 2. Single movie page with its own poster
<img width="1374" alt="Screenshot 2024-06-29 at 12 51 29 AM" src="https://media.github.khoury.northeastern.edu/user/17515/files/6aeb5c2f-5cee-4f29-929b-e0aa854707af">


### 3. Single movie page with the placeholder poster
<img width="1335" alt="Screenshot 2024-06-29 at 1 27 11 AM" src="https://media.github.khoury.northeastern.edu/user/17515/files/f93172c9-0e38-41ca-b7f8-5d4986ec5b0c">


### 4. Command line showing passed tests
<img width="380" alt="Screenshot 2024-06-29 at 1 28 39 AM" src="https://media.github.khoury.northeastern.edu/user/17515/files/fa88b8d2-6bf4-4be0-8f3d-665b4efb0000">

<img width="375" alt="Screenshot 2024-06-29 at 1 28 34 AM" src="https://media.github.khoury.northeastern.edu/user/17515/files/74bf3159-7084-4343-90c3-b5ba781283ff">

<img width="375" alt="Screenshot 2024-06-29 at 1 28 44 AM" src="https://media.github.khoury.northeastern.edu/user/17515/files/4d8d2547-503f-4c10-915e-262ae479221f">

### Code Splitting

## What Was Done
For this assignment, the following tasks were completed:
1. **Set up the frontend repository** to display movies and their details.
2. **Implemented a mock server** using `msw` (Mock Service Worker) to simulate API responses for testing purposes.
3. **Developed and updated components**:
   - `MoviesList` component to display a list of movies.
   - `Movie` component to display detailed information about a single movie.
4. **Added tests** to verify that components correctly render the expected data:
   - Tests for `MoviesList` to ensure it displays the correct number of movie cards.
   - Tests for `Movie` to ensure it displays the correct movie details and reviews.
5. **Configured the environment**:
   - Set up environment variables for API base URL.
   - Ensured compatibility with Node.js by addressing issues such as `TextEncoder` not being defined.
