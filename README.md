# Next Word Probability Engine

A FastAPI web application that predicts the most probable next words in a sequence using Google's Gemini AI, featuring an interactive neural network visualization.

![Screenshot](https://via.placeholder.com/800x500.png?text=Next+Word+Probability+Engine+Screenshot)

## Features

- **AI-Powered Predictions**: Uses Google's Gemini model to predict next word probabilities
- **Interactive Visualization**: Real-time neural network animation showing text processing
- **Probability Charts**: Visual representation of top predicted words and their probabilities
- **Responsive Design**: Works on desktop and mobile devices
- **Mathematical Foundation**: Built on probabilistic language modeling principles

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/next-word-predict.git
cd next-word-predict
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -e .
```

4. Set up environment variables:
```bash
cp .env.example .env
```

5. Obtain a [Google Gemini API key](https://ai.google.dev/) and add it to your `.env` file:
```env
GEMINI_API_KEY=your_api_key_here
```

## Usage

1. Start the development server:
```bash
uvicorn main:app --reload
```

2. Open your browser to [http://localhost:8000](http://localhost:8000)

3. Enter text in the input box and click "Predict Next Word" to see:
   - Top 5 predicted next words with probabilities
   - Interactive neural network visualization
   - Probability distribution chart

## Project Structure

```
next-word-predict/
├── main.py               # FastAPI application and prediction logic
├── pyproject.toml        # Project dependencies and metadata
├── README.md             # This documentation
├── static/               # Static files (CSS, JS, images)
├── templates/            # HTML templates
│   └── index.html        # Main application interface
└── .env                  # Environment configuration
```

## Dependencies

- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Google Generative AI](https://ai.google.dev/) - Language model API
- [Jinja2](https://jinja.palletsprojects.com/) - Templating engine
- [Uvicorn](https://www.uvicorn.org/) - ASGI server
- [Chart.js](https://www.chartjs.org/) - Frontend charting (included via CDN)
- [MathJax](https://www.mathjax.org/) - Mathematical typesetting (included via CDN)

## Configuration

The application can be configured via environment variables:

- `GEMINI_API_KEY`: Required - Your Google Gemini API key
- `GEMINI_MODEL`: Optional - Defaults to "gemini-1.5-flash"
- `PORT`: Optional - Server port (default: 8000)

## Development

To run in development mode with auto-reload:
```bash
uvicorn main:app --reload
```

For production deployment, consider using:
- [Gunicorn](https://gunicorn.org/) with Uvicorn workers
- Docker containerization
- Cloud deployment (AWS, GCP, Azure)

## License

MIT License - See [LICENSE](LICENSE) for details.
