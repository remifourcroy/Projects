\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{listings}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{xcolor}
\usepackage{fancyvrb}
\usepackage{upquote}

\title{\textbf{Custom Sobel Filter Implementation}}
\author{}
\date{}

\begin{document}

\maketitle

\section*{Overview}

This document provides an implementation of the \textbf{Sobel filter}, a popular edge-detection algorithm used in image processing. The Sobel filter computes the gradient magnitude in both horizontal and vertical directions to identify edges in images.

We have implemented two versions:

\begin{enumerate}
    \item \textbf{Using \texttt{scipy.signal.convolve2d}}: A concise and efficient implementation leveraging SciPy's optimized convolution function.
    \item \textbf{Without external libraries for convolution}: A detailed, manual implementation using nested loops to compute the convolution operation step by step.
\end{enumerate}

\section*{Features}

\begin{itemize}
    \item Apply the Sobel filter to grayscale or color images.
    \item Compute edge detection with horizontal and vertical gradients.
    \item Two implementations available:
    \begin{itemize}
        \item \textbf{Efficient implementation}: Uses \texttt{convolve2d} from SciPy for convolution.
        \item \textbf{Manual implementation}: Performs convolution using nested loops for learning purposes.
    \end{itemize}
    \item Handles edge pixels using \textbf{padding techniques} (e.g., reflective padding).
\end{itemize}

\section*{Requirements}

\begin{itemize}
    \item Python 3.x
    \item Required libraries:
    \begin{itemize}
        \item \texttt{numpy}
        \item \texttt{scipy}
        \item \texttt{opencv-python}
        \item \texttt{matplotlib} (for visualization)
    \end{itemize}
\end{itemize}

Install the required libraries with:

\begin{Verbatim}[commandchars=\\\{\}]
pip install numpy scipy opencv-python matplotlib
\end{Verbatim}

\section*{Usage}

\subsection*{1. Clone the Repository}

\begin{Verbatim}[commandchars=\\\{\}]
git clone https://github.com/yourusername/sobel-filter.git
cd sobel-filter
\end{Verbatim}

\subsection*{2. Run the Script}

Make sure you have an image (e.g., \texttt{image.jpg}) in the project directory. Then, execute the script:

\begin{Verbatim}[commandchars=\\\{\}]
python sobel_filter.py
\end{Verbatim}

\subsection*{3. Outputs}

The script:

\begin{itemize}
    \item Loads the input image (grayscale or color).
    \item Applies both Sobel implementations (with and without \texttt{convolve2d}).
    \item Displays the original image and the resulting edge-detected images.
\end{itemize}

\section*{Project Structure}

\begin{Verbatim}[commandchars=\\\{\}]
sobel-filter/
├── sobel_filter.py      # Main implementation and testing script
├── README.md            # Project documentation
└── image.jpg            # Example input image (replace with your own)
\end{Verbatim}

\section*{Key Functions}

\subsection*{1. \texttt{sobel\_filter\_custom(image)}}

\begin{itemize}
    \item Implements the Sobel filter \textbf{manually}, without using external convolution functions.
    \item Uses \textbf{nested loops} to perform convolution with the Sobel kernels.
\end{itemize}

\subsection*{2. \texttt{sobel\_filter\_convolve2d(image)}}

\begin{itemize}
    \item Implements the Sobel filter \textbf{using \texttt{scipy.signal.convolve2d}}.
    \item Simpler and faster, leveraging optimized convolution.
\end{itemize}

\section*{References}

\begin{itemize}
    \item \href{https://en.wikipedia.org/wiki/Sobel_operator}{Sobel Filter}: Learn more about the Sobel operator and its applications in edge detection.
    \item \href{https://numpy.org/doc/stable/reference/generated/numpy.pad.html}{Padding Image Functions}: Documentation on NumPy's \texttt{pad} function for handling image boundaries.
    \item \href{https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html}{SciPy \texttt{convolve2d}}: Detailed explanation of the \texttt{convolve2d} function used for convolution.
    \item \href{https://docs.opencv.org/master/d6/d00/tutorial_py_root.html}{OpenCV Documentation}: Guide to loading and manipulating images in Python.
\end{itemize}

\section*{Example Results}

\subsection*{Input Image}

\begin{center}
\includegraphics[width=0.6\textwidth]{original_image.png}
\end{center}

\subsection*{Output (Edge Detection)}

\begin{enumerate}
    \item \textbf{Using \texttt{convolve2d}:}

    \begin{center}
    \includegraphics[width=0.6\textwidth]{edges_convolve2d.png}
    \end{center}

    \item \textbf{Using Nested Loops:}

    \begin{center}
    \includegraphics[width=0.6\textwidth]{edges_manual.png}
    \end{center}
\end{enumerate}

\section*{Why Two Implementations?}

This project provides both implementations to balance:

\begin{itemize}
    \item \textbf{Efficiency}: The \texttt{convolve2d} implementation is ideal for practical use due to its speed and simplicity.
    \item \textbf{Learning}: The manual implementation breaks down convolution step by step, helping beginners understand how it works under the hood.
\end{itemize}

\section*{Contributing}

Feel free to open issues or submit pull requests for:

\begin{itemize}
    \item Enhancements to the code.
    \item Adding new features (e.g., other filters).
    \item Suggestions for improvement.
\end{itemize}

\section*{License}

This project is licensed under the MIT License. See \texttt{LICENSE} for details.

\end{document}
