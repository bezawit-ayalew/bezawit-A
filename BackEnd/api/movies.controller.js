import MoviesDAO from '../dao/moviesDAO.js';

export default class MoviesController {
    static async apiGetMovies(req, res, next) {
        try {
            const moviesPerPage = req.query.moviesPerPage ? parseInt(req.query.moviesPerPage, 10) : 20;
            const page = req.query.page ? parseInt(req.query.page, 10) : 0;

            let filters = {};
            if (req.query.rated) {
                filters.rated = req.query.rated;
            } else if (req.query.title) {
                filters.title = req.query.title;
            }

            const { moviesList, totalNumMovies } = await MoviesDAO.getMovies({
                filters,
                page,
                moviesPerPage,
            });

            let response = {
                movies: moviesList,
                page: page,
                filters: filters,
                entries_per_page: moviesPerPage,
                total_results: totalNumMovies,
            };
            res.json(response);
        } catch (e) {
            console.error(`Something went wrong in apiGetMovies: ${e}`);
            res.status(500).json({ error: e });
        }
    }

    static async apiGetMovieById(req, res, next) {
        try {
            let id = req.params.id || {};
            let movie = await MoviesDAO.getMovieById(id);
            if (!movie) {
                res.status(404).json({ error: "not found" });
                return;
            }
            res.json(movie);
        } catch (e) {
            console.error(`Something went wrong in apiGetMovieById: ${e}`);
            res.status(500).json({ error: e });
        }
    }

    static async apiGetRatings(req, res, next) {
        try {
            let ratings = await MoviesDAO.getRatings();
            res.json(ratings);
        } catch (e) {
            console.error(`Something went wrong in apiGetRatings: ${e}`);
            res.status(500).json({ error: e });
        }
    }
}

