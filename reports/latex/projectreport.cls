% Template for Semester Project Report at KU DoCSE
\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{projectreport}[2022/03/27]
\usepackage{hyperref}

\LoadClass[12pt]{report}

\RequirePackage[a4paper,left=1.5in,right=1.25in,top=1.25in,bottom=1.25in]{geometry}

% Use Times New Roman font
\RequirePackage{times}

\RequirePackage{graphicx}
\RequirePackage{sectsty}

% For Bibliography
\RequirePackage{natbib}
\bibliographystyle{apalike}
\renewcommand{\bibname}{References} 

% For abbreviations / acronyms
\usepackage{glossaries}
\loadglsentries{glossaries}
\makenoidxglossaries

% For tables
\usepackage{tabularx} 
\newcolumntype{Y}{>{\raggedright\arraybackslash}X} % left-aligned wrapped text

% All the figures should be in figures directory
\graphicspath{{figures/}}

% Year
\global\let\@y\@empty
\def\y#1{%
	\ifx\@y\@empty
		\gdef\@y{#1}
	\fi}

% Semester / Part
\global\let\@sem\@empty
\def\sem#1{%
	\ifx\@sem\@empty
		\gdef\@sem{#1}
	\fi}

% Group (CE/CS)
\global\let\@group\@empty
\def\group#1{%
	\ifx\@group\@empty
		\gdef\@group{#1}
	\fi}

% Course code
\global\let\@coursecode\@empty
\def\coursecode#1{%
	\ifx\@coursecode\@empty
		\gdef\@coursecode{#1}
	\fi}

% Coordinator's name
\global\let\@coordinator\@empty
\def\coordinator#1{%
	\ifx\@coordinator\@empty
	\gdef\@coordinator{#1}
	\fi}

% Supervisor details
\global\let\@supervisorname\@empty
\global\let\@supervisortitle\@empty
\global\let\@supervisordept\@empty
\newcommand\supervisor[3]{%
	\ifx\@supervisorname\@empty
		\gdef\@supervisorname{#1}
	\fi
	\ifx\@supervisortitle\@empty
		\gdef\@supervisortitle{#2}
	\fi
	\ifx\@supervisordept\@empty
		\gdef\@supervisordept{#3}
	\fi}

% Project team members
\global\let\@firstmember\@empty
\global\let\@secondmember\@empty
\global\let\@thirdmember\@empty
\global\let\@fourthmember\@empty
\global\let\@fifthmember\@empty
\global\let\@sixthmember\@empty

\newcommand\member[2]{%
	\ifx\@firstmember\@empty
		\gdef\@firstmember{#1 (#2)}
	\else\ifx\@secondmember\@empty
		\gdef\@secondmember{#1 (#2)}
	\else\ifx\@thirdmember\@empty
		\gdef\@thirdmember{#1 (#2)}
	\else\ifx\@fourthmember\@empty
		\gdef\@fourthmember{#1 (#2)}
	\else\ifx\@fifthmember\@empty
		\gdef\@fifthmember{#1 (#2)}
	\else\ifx\@sixthmember\@empty
		\gdef\@sixthmember{#1 (#2)}
	\else
		\ClassError{projectreport}{Too many members}{}%
	\fi\fi\fi\fi\fi\fi}

% Font size for Section, Subsection, Chapter
\sectionfont{\fontsize{14}{20}\selectfont}
\subsectionfont{\fontsize{13}{18}\selectfont}
\subsubsectionfont{\fontsize{12}{17}\selectfont}
\chapterfont{\fontsize{16}{24}\selectfont}


% Customize Chapter heading (Centered, two-line format)
\def\@makechapterhead#1{%
  
  \begin{center}
    {\fontsize{16}{24}\fontseries{b}\selectfont
    Chapter \thechapter\par}
    \vspace{6pt}
    {\fontsize{16}{24}\fontseries{b}\selectfont
    #1\par}
  \end{center}
  \vspace{20pt}
}

% \makeatletter
\def\@makeschapterhead#1{%
  \begin{center}
    {\fontsize{16}{24}\fontseries{b}\selectfont
    #1\par}
  \end{center}
  \vspace{20pt}
}
% \makeatother



% Title page
\renewcommand{\maketitle}{
	\begin{titlepage}
		\centering
        \includegraphics[width=4cm]{logo-200}
        \vspace{0.5cm}
        
		\fontseries{b}
		\fontsize{16}{24}
		\selectfont
		
		Kathmandu University
				
		Department of Computer Science and Engineering 
		
		Dhulikhel, Kavre
		
		
		
		\vspace{1cm}
		
		A Project Report 
		
		on
		
		``\@title''
		
		\vspace{1cm}
		[Code No.: \@coursecode]
		
		\fontsize{12}{18}\selectfont
		(For partial fulfillment of Year \@y{} / Semester \@sem{} in \@group)
		
		\fontsize{14}{21}\selectfont
		\vspace{1.2cm}
		
		Submitted by
		
		\fontsize{12}{18}\selectfont
		\ifx\@firstmember\@empty
		\else
			\@firstmember
		\fi
		
		\ifx\@secondmember\@empty
		\else
			\@secondmember
		\fi
		
		\ifx\@thirdmember\@empty
		\else
			\@thirdmember
		\fi
		
		\ifx\@fourthmember\@empty
			\phantom{4}
		\else
			\@fourthmember
		\fi
		
		\ifx\@fifthmember\@empty
			\phantom{5}
		\else
			\@fifthmember
		\fi
		
		\ifx\@sixthmember\@empty
			
		\else
			\@sixthmember
		\fi
		
		\vspace{1cm}
		\fontsize{14}{21}\selectfont
		
		Submitted to
		
		\fontsize{12}{18}\selectfont
		\@coordinator
		
		Department of Computer Science and Engineering
		
		\vspace{0.5cm}
		\@date
		
	\end{titlepage}
}

% Bona fide Certificate
\newcommand{\makebonafidecertificate}{
	\newpage
	\thispagestyle{empty}
	\newgeometry{top=2in}
	\begin{center}
	{
		\centering
		\fontseries{b}\fontsize{16}{24}\selectfont
		Bona fide Certificate
		\vspace{2cm}
		
		\fontseries{b}\fontsize{14}{28}\selectfont
		This project work on
		
		``\@title''
		
		is the bona fide work of
		
		``\\
		\ifx\@firstmember\@empty
		\else
		\@firstmember
		\fi
		\ifx\@secondmember\@empty
		\else
		,\\ \@secondmember
		\fi
		\ifx\@thirdmember\@empty
		\else
		,\\ \@thirdmember
		\fi
		\ifx\@fourthmember\@empty
		\else
		,\\ \@fourthmember
		\fi
		\ifx\@fifthmember\@empty
		\else
		,\\  \@fifthmember
		\fi
		\ifx\@sixthmember\@empty
		\else
		,\\ \@sixthmember
		\fi
		\\
		''\\
		
		who carried out the project work under my supervision.}
	
	\end{center}
	\vspace{2cm}
	{
		\fontseries{b}\fontsize{14}{20}\selectfont

		Project Supervisor
		
		\vspace{2cm}

		\@supervisorname

		\@supervisortitle
		
		\@supervisordept
	}
	\restoregeometry
	
}

% Abstract 
\renewenvironment{abstract}
{
	\newpage
	\setlength{\parindent}{0pt} % <-- NO indentation in abstract
	\fontseries{b}
	\fontsize{16}{24}
	\selectfont
	\begin{center}
		Abstract
	\end{center}

	\normalfont\fontsize{12}{18}\selectfont
}
{
	\par
}

% Keywords
\newenvironment{keywords}
{
    \vspace{0.5cm}
	\noindent
	\normalfont\fontsize{12}{18}\selectfont
    % \hspace*{0.5cm}
	\textbf{Keywords:}  
	\fontshape{it}\selectfont
}

% Acknowledgements
\newenvironment{acknowledgements}
{
	\newpage
	% \thispagestyle{empty}
	\fontseries{b}\fontsize{16}{24}	\selectfont
	\begin{flushleft}
		\centering\textbf{Acknowledgements}\par
	\end{flushleft}
	
	\normalfont\fontsize{12}{18}\selectfont
	\noindent
}




