import React, { useState, useEffect } from 'react';
import MovieDataService from '../services/movies';
import { Link, useParams } from 'react-router-dom';
import Card from 'react-bootstrap/Card';
import Container from 'react-bootstrap/Container';
import Image from 'react-bootstrap/Image';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Button from 'react-bootstrap/Button';


const Movie = ({ user }) => {
  const [movie, setMovie] = useState({});
  const params = useParams();

  useEffect(() => {
    MovieDataService.get(params.id)
      .then(response => {
        setMovie(response.data);
      })
      .catch(e => {
        console.log(e);
      });
  }, [params.id]);

  const deleteReview = (reviewId, index) => {
    let data = {
      review_id: reviewId,
      user_id: user.googleId
    };
  
    MovieDataService.deleteReview(data)
      .then(response => {
        setMovie((prevState) => {
          prevState.reviews.splice(index, 1);
          return {
            ...prevState
          };
        });
      })
      .catch(e => {
        console.log(e);
      });
  };
  

  return (
    <Container>
      <Row>
        <Col>
          <Image src={movie.poster} alt="movie poster" />
        </Col>
        <Col>
          <Card>
            <Card.Body>
              <Card.Title>{movie.title}</Card.Title>
              <Card.Text>{movie.plot}</Card.Text>
              {user && (
                <Link to={"/movies/" + params.id + "/review"}>
                  Add Review
                </Link>
              )}
              {user && movie.reviews && movie.reviews.map((review, index) => (
                <Card key={index}>
                  <Card.Body>
                    <Card.Text>{review.review}</Card.Text>
                    <Button variant="link" onClick={() => deleteReview(review._id, index)}>
                      Delete
                    </Button>
                    <Link
                      to={{
                        pathname: "/movies/" + params.id + "/review",
                        state: { currentReview: review }
                      }}
                    >
                      Edit
                    </Link>
                  </Card.Body>
                </Card>
              ))}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  )
}

export default Movie;
