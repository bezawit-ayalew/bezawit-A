import supertest from 'supertest';
import { expect } from 'chai';
import app from '../server.js';

const request = supertest(app);

describe('Testing Movie and Review API endpoints', function () {
    this.timeout(10000);

    it('GET /api/v1/movies should return a list of movies', async function () {
        const response = await request.get('/api/v1/movies');
        expect(response.status).to.equal(200);
        expect(response.body.movies).to.be.an('array');
    });

    it('GET /api/v1/movies/id/:id should return a movie by ID', async function () {
        const movieId = '573a1390f29313caabcd4135'; // Replace with a valid movie ID from your database
        const response = await request.get(`/api/v1/movies/id/${movieId}`);
        expect(response.status).to.equal(200);
        expect(response.body).to.be.an('object');
        expect(response.body._id).to.equal(movieId);
    });

    it('GET /api/v1/movies/ratings should return a list of ratings', async function () {
        const response = await request.get('/api/v1/movies/ratings');
        expect(response.status).to.equal(200);
        expect(response.body).to.be.an('array');
    });

    it('POST /api/v1/movies/review should post a review', async function () {
        const review = {
            movie_id: '573a1390f29313caabcd4135', // Replace with a valid movie ID from your database
            review: 'This movie is OK by me!',
            user_id: '1234',
            name: 'Jane Doe'
        };
        const response = await request.post('/api/v1/movies/review').send(review);
        expect(response.status).to.equal(200);
        expect(response.body.status).to.equal('success');
    });

    it('PUT /api/v1/movies/review/:id should update a review', async function () {
        const updatedReview = {
            review_id: 'some_review_id', // Replace with a valid review ID from your database
            review: 'Actually, after giving it a little thought I think this is actually a bad movie.',
            user_id: '1234',
            name: 'Jane Doe'
        };
        const response = await request.put(`/api/v1/movies/review/some_review_id`).send(updatedReview);
        expect(response.status).to.equal(200);
        expect(response.body.status).to.equal('success');
    });

    it('DELETE /api/v1/movies/review/:id should delete a review', async function () {
        const deleteReview = {
            review_id: 'some_review_id', // Replace with a valid review ID from your database
            user_id: '1234'
        };
        const response = await request.delete(`/api/v1/movies/review/some_review_id`).send(deleteReview);
        expect(response.status).to.equal(200);
        expect(response.body.status).to.equal('success');
    });
});

