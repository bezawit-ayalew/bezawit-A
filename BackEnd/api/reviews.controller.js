import ReviewsDAO from '../dao/reviewsDAO.js';

export default class ReviewsController {
    static async apiPostReview(req, res, next) {
        try {
            const movieId = req.body.movie_id;
            const review = req.body.review;
            const userInfo = {
                name: req.body.name,
                _id: req.body.user_id,
            };
            const date = new Date();

            const reviewResponse = await ReviewsDAO.addReview(
                movieId,
                userInfo,
                review,
                date,
            );
            var { error } = reviewResponse;
            if (error) {
                res.status(500).json({ error: "Unable to post review." });
            } else {
                res.json({ status: "success", response: reviewResponse });
            }
        } catch (e) {
            console.error(`Something went wrong in apiPostReview: ${e}`);
            res.status(500).json({ error: e.message });
        }
    }

    static async apiUpdateReview(req, res, next) {
        try {
            const reviewId = req.params.id;
            const review = req.body.review;
            const date = new Date();

            const reviewResponse = await ReviewsDAO.updateReview(
                reviewId,
                review,
                date,
            );
            var { error } = reviewResponse;
            if (error) {
                res.status(500).json({ error: "Unable to update review." });
            } else if (reviewResponse.modifiedCount === 0) {
                throw new Error("Unable to update review - review not found.");
            } else {
                res.json({ status: "success", response: reviewResponse });
            }
        } catch (e) {
            console.error(`Something went wrong in apiUpdateReview: ${e}`);
            res.status(500).json({ error: e.message });
        }
    }

    static async apiDeleteReview(req, res, next) {
        try {
            const reviewId = req.params.id;
            const userId = req.body.user_id;

            const reviewResponse = await ReviewsDAO.deleteReview(
                reviewId,
                userId,
            );
            var { error } = reviewResponse;
            if (error) {
                res.status(500).json({ error: "Unable to delete review." });
            } else if (reviewResponse.deletedCount === 0) {
                throw new Error("Unable to delete review - review not found.");
            } else {
                res.json({ status: "success", response: reviewResponse });
            }
        } catch (e) {
            console.error(`Something went wrong in apiDeleteReview: ${e}`);
            res.status(500).json({ error: e.message });
        }
    }
}

