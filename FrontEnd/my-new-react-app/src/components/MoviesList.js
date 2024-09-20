import React, { useState, useEffect, useCallback } from 'react';
import MovieDataService from '../services/movies';
import { Link } from 'react-router-dom';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Container from 'react-bootstrap/Container';
import Card from 'react-bootstrap/Card';
import './MoviesList.css';
import { BsStar, BsStarFill } from "react-icons/bs";


const MoviesList = ({
    user,
    favorites,
    addFavorite,
    deleteFavorite
}) => {
    // component implementation

    const [movies, setMovies] = useState([]);
    const [searchTitle, setSearchTitle] = useState('');
    const [searchRating, setSearchRating] = useState('');
    const [ratings, setRatings] = useState(['All Ratings']);
    const [currentPage, setCurrentPage] = useState(1);
    const [entriesPerPage, setEntriesPerPage] = useState(10); // Assuming a default value
    const [currentSearchMode, setCurrentSearchMode] = useState('');

    const retrieveRatings = useCallback(() => {
        MovieDataService.getRatings()
            .then(response => {
                console.log("Ratings response:", response.data);
                setRatings(['All Ratings'].concat(response.data));
            })
            .catch(e => {
                console.log(e);
            });
    }, []);
    const retrieveMovies = useCallback(() => {
        setCurrentSearchMode("");
        MovieDataService.getAll(currentPage)
            .then(response => {
                setMovies(response.data.movies);
                setCurrentPage(response.data.page);
                setEntriesPerPage(response.data.entries_per_page);
            })
            .catch(e => {
                console.log(e);
            });
    }, [currentPage]);

    const find = useCallback((query, by, page) => {
        MovieDataService.find(query, by, page)
            .then(response => {
                console.log("Movies response:", response.data);
                setMovies(response.data.movies);
            })
            .catch(e => {
                console.log(e);
            });
    }, []);

    const findByTitle = useCallback(() => {
        setCurrentSearchMode("findByTitle");
        find(searchTitle, "title", currentPage);
    }, [find, searchTitle, currentPage]);

    const findByRating = useCallback(() => {
        setCurrentSearchMode("findByRating");
        if (searchRating === "All Ratings") {
            retrieveRatings(); // This might be intended to be another function like retrieveMovies
        } else {
            find(searchRating, "rated", currentPage);
        }
    }, [find, searchRating, currentPage, retrieveRatings]);
    const retrieveNextPage = useCallback(() => {
        if (currentSearchMode === "findByTitle") {
            findByTitle();
        } else if (currentSearchMode === "findByRating") {
            findByRating();
        } else {
            retrieveMovies();
        }
    }, [currentSearchMode, findByTitle, findByRating, retrieveMovies]);


    // Use effect to carry out side-effect functionality
    useEffect(() => {
        retrieveRatings();
    }, [retrieveRatings]);

    useEffect(() => {
        setCurrentPage(0);
    }, [currentSearchMode]);


    useEffect(() => {
        retrieveNextPage();
    }, [currentPage, retrieveNextPage]);



    const onChangeSearchTitle = e => {
        setSearchTitle(e.target.value);
    };

    const onChangeSearchRating = e => {
        setSearchRating(e.target.value);
    };

    return (
        <div className="App">
            <Container className="main-container">
                <Form>
                    <Row>
                        <Col>
                            <Form.Group className="mb-3">
                                <Form.Control
                                    type="text"
                                    placeholder="Search by title"
                                    value={searchTitle}
                                    onChange={onChangeSearchTitle}
                                />
                            </Form.Group>
                            <Button variant="primary" type="button" onClick={findByTitle}>
                                Search
                            </Button>
                        </Col>
                        <Col>
                            <Form.Group className="mb-3">
                                <Form.Control as="select" value={searchRating} onChange={onChangeSearchRating}>
                                    {ratings.map((rating, i) => (
                                        <option value={rating} key={i}>
                                            {rating}
                                        </option>
                                    ))}
                                </Form.Control>
                            </Form.Group>
                            <Button variant="primary" type="button" onClick={findByRating}>
                                Search
                            </Button>
                        </Col>
                    </Row>
                </Form>
                <Row className="movieRow">
                    {movies.length > 0 ? movies.map(movie => (
                        <Col key={movie._id}>
                            <Card className="moviesListCard">
                                {user && (
                                    favorites.includes(movie._id) ?
                                        <BsStarFill className="star starFill" onClick={() => {
                                            deleteFavorite(movie._id);
                                        }} />
                                        :
                                        <BsStar className="star starEmpty" onClick={() => {
                                            addFavorite(movie._id);
                                        }} />
                                )}

                                <Card.Img
                                    className="smallPoster"
                                    src={movie.poster + "/100px180"}
                                    onError={({ currentTarget }) => {
                                        currentTarget.onerror = null; // prevents looping
                                        currentTarget.src = "/images/NoPosterAvailable-crop.jpg";
                                    }}
                                />
                                <Card.Body>
                                    <Card.Title>{movie.title}</Card.Title>
                                    <Card.Text>Rating: {movie.rated}</Card.Text>
                                    <Card.Text>{movie.plot}</Card.Text>
                                    <Link to={"/movies/" + movie._id}>View Reviews</Link>
                                </Card.Body>
                            </Card>
                        </Col>
                    )) : <div>No movies found</div>}
                </Row>
                <br />
                Showing page: {currentPage + 1}.
                <Button
                    variant="link"
                    onClick={() => { setCurrentPage(currentPage + 1) }}>
                    Get next {entriesPerPage} results
                    {/* Showing page: {currentPage}.
                <Button variant="link" onClick={() => setCurrentPage(prevPage => prevPage + 1)}>
                    Get next {entriesPerPage} results */}
                </Button>
            </Container>
        </div>
    );
}

export default MoviesList;
