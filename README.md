# Life Expectancy Prediction Model using Flask and Heroku

### Using ML model to predict Life Expectancy for each country and how to deploy a Python Flask application to Heroku.

### 3 options for running the project:
1. **Local run for testing** - run as a python file `server.py` without Gunicorn locally.  Since environment variable `'PORT'` does not exist, will listen on default Flask port `5000` on `localhost` only.
2. **Heroku without Gunicorn** - run as a python file `server.py` without Gunicorn on Heroku.  Use the following line in `Procfile`:
   `web: python server.py`.  Will listen on port given by Heroku in environment variable `PORT`
3. **Heroku with Gunicorn** - change `Procfile` file to: 
   `web: gunicorn server:app`.
   Then Gunicorn will run inside Heroku dyno with multiple instances of the `server.py` specified by Heroku in environment variable `WEB_CONCURRENCY` depending on available and used memory in the dyno.
   