 

## Frontend Repository

### Screenshot of the Top Page with Favorites Stars Displayed
![image](https://github.com/user-attachments/assets/bc461e8d-e465-4977-8772-2706e39b1a73)

### Screenshot of the Favorites List Database Table Displayed in MongoDB Compass
 
# Movie Review App - Part 4: Authentication


## Overview

In this part of the project, I set up authentication to enable users to log in and log out, ensuring that certain functionalities (specifically the ability to create, edit, and delete reviews) are available only when users are logged in. I used Google's authentication API to handle this. Anyone with a Gmail account will automatically be able to identify themselves to our application.
## Available Scripts


1. A screenshot of a movie reviews page with a newly written review:

![image](https://github.com/user-attachments/assets/5c7007bd-b536-4f92-874e-46430d03b52c)
   



2. A screenshot of the same page with the review edited:

 


# Movie App Assignment - Part 3 

## Description
This repository contains the frontend for a Movie App. The app allows users to browse a list of movies, view detailed information about each movie, and see reviews. This assignment focuses on demonstrating the integration of a mock server for testing purposes and ensuring that the frontend correctly renders data.


### 1. App at localhost:3000 with at least one movie showing the placeholder poster
The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

![image](https://github.com/user-attachments/assets/8829c8ec-9317-4dcb-8f50-d72b51b15ec4)

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### 2. Single movie page with its own poster

![image](https://github.com/user-attachments/assets/9ec8f248-eba5-4e28-b3ef-75310826ee78)



### 3. Single movie page with the placeholder poster

![image](https://github.com/user-attachments/assets/ce2ce8bb-6497-4fb5-9e4f-9ddfb0803122)



### 4. Command line showing passed tests
![image](https://github.com/user-attachments/assets/ed3a0fb0-fd3f-47aa-8d7b-925ee1456b90)

![image](https://github.com/user-attachments/assets/9dda6e7f-de14-4d0a-99fa-5ce0e6926e09)

![image](https://github.com/user-attachments/assets/e173bcd8-0030-42f2-9fd8-517b6578fc99)


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
