import mongodb from 'mongodb';
import dotenv from 'dotenv';
import app from './server.js';
import MoviesDAO from './dao/moviesDAO.js';
import ReviewsDAO from './dao/reviewsDAO.js';
import FavoritesDAO from './dao/favoritesDAO.js';

async function main() {
    dotenv.config({ path: 'env.env' });

    const client = new mongodb.MongoClient(process.env.MOVIEREVIEWS_DB_URI, {
        useNewUrlParser: true,
        useUnifiedTopology: true,
    });

    const port = process.env.PORT || 8000;

    console.log("MongoDB URI:", process.env.MOVIEREVIEWS_DB_URI);

    try {
        await FavoritesDAO.injectDB(client);
        console.log("Connected to MongoDB");

        await MoviesDAO.injectDB(client);
        await ReviewsDAO.injectDB(client);

        app.listen(port, () => {
            console.log(`Server is running on port ${port}`);
        });
    } catch (e) {
        console.error("Failed to connect to MongoDB", e);
        process.exit(1);
    }
}

main().catch(console.error);

export default app;

