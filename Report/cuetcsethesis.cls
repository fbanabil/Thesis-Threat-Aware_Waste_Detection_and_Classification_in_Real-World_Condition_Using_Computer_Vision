% CUET CSE THESIS FINAL REPORT FORMAT @version     1.0
% Get in touch if you have any questions or problems!
% <atiqul.rizvi@cuet.ac.bd>

% Class file to define structures of thesis report. Modification to any part may
% result in producing errors. Don't try to modify unless absolutely necessary and 
% without prior knowledge and consequences of such modification.

% defining variables as long globals
\def\lxdef{\long\xdef}

% We need LaTeX2e at least
\NeedsTeXFormat{LaTeX2e}

% Article class
\LoadClass[12pt, a4paper]{report}


% PACKAGES
\RequirePackage{amsmath}
\RequirePackage{ifthen}
\RequirePackage{xkeyval}
\RequirePackage[dvipsnames]{xcolor}
\RequirePackage[T1]{fontenc}
\RequirePackage[UKenglish]{babel}
\RequirePackage[UKenglish]{isodate}
\RequirePackage[utf8]{inputenc}
\RequirePackage{xparse}
\RequirePackage{graphicx}
\RequirePackage{longtable}
\RequirePackage[acronym,toc]{glossaries}
\RequirePackage{multirow}
\RequirePackage{blindtext}
\RequirePackage{parskip}
\RequirePackage{setspace}
\RequirePackage{geometry}
\RequirePackage{layout}
\RequirePackage{courier}
\RequirePackage{caption}
\RequirePackage{subcaption}
\captionsetup[figure]{font=footnotesize}
\RequirePackage{url}
\RequirePackage{hyperref}
\RequirePackage[table,figure]{totalcount}
\RequirePackage{appendix}
\RequirePackage{csquotes}
\RequirePackage{titlesec}
\RequirePackage{titletoc}
\RequirePackage{datetime2}
\RequirePackage{fancyhdr}
\RequirePackage{lmodern}
\RequirePackage{etoolbox}
\RequirePackage{cleveref}


\usepackage{graphicx}
\usepackage{float}
\usepackage{placeins}


% DATE
\newcommand{\monthyeardate}{%
  \DTMenglishmonthname{\@dtm@month} \@dtm@year
}

\newcommand{\yeardate}{%
  \@dtm@year
}

% Table Font Declaration
\let\oldtabular\tabular
\renewcommand{\tabular}{\footnotesize\oldtabular}
\captionsetup{font=footnotesize}

% THESIS TEMPLATE VARIABLES
% -------------------------
% Basic information regarding the student, thesis, department and university
\lxdef \@logo           {resources/logo}
\newcommand{\thesisLogoPath}[1]{\xdef\@logo{#1}}
\lxdef \@date           {\today}% safe default; overridden by \thesisSubmissionDate
\newbool{bauth}
\newcommand{\authorB}[1]{\xdef\@authorB{#1} \booltrue{bauth}}
\lxdef \@authorB         {}
\newcommand{\studentBNumber}[1]{\xdef\@IDB{#1}}
\lxdef \@IDB         {}
\newcommand{\thesisSubmissionDate}[1]{\xdef\@date{#1}}
\lxdef \@degree         {Bachelor of Science}
\newcommand{\thesisDegree}[1]{\xdef\@degree{#1}}
\lxdef \@shortdegree         {B.Sc.}
\newcommand{\shortformofDegree}[1]{\xdef\@shortdegree{#1}}
\lxdef \@department     {Computer Science \& Engineering}
\newcommand{\thesisdepartment}[1]{\xdef\@department{#1}}
\lxdef \@universityname {Chittagong University of Engineering \& Technology (CUET)}
\newcommand{\thesisUniversity}[1]{\xdef\@universityname{#1}}

% Student info -----
\newcommand{\authorA}[1]{\xdef\@authorA{#1}}
\newcommand{\studentANumber}[1]{\xdef\@IDA{#1}}
\newcommand{\studentsession}[1]{\xdef\@studentsession{#1}}

% Supervisor info ------
\newcommand{\thesisSupervisor}[1]{\xdef\@supervisor{#1}}
\newcommand{\thesisSupervisorDesignation}[1]{\xdef\@supervisordesignation{#1}}

% External Supervisor info ------
\newcommand{\thesisExternalSupervisor}[1]{\xdef\@externalsupervisor{#1}}
\newcommand{\thesisExternalSupervisorDesignation}[1]{\xdef\@externaldesignation{#1}}
\newcommand{\thesisExternalSupervisorDepartment}[1]{\xdef\@externaldepartment{#1}}
\newcommand{\thesisExternalSupervisorUniversity}[1]{\xdef\@externaluniversity{#1}}

% Department Head info ------
\newcommand{\thesisDepartmentHead}[1]{\xdef\@departmenthead{#1}}
\newcommand{\thesisDepartmentHeadDesignation}[1]{\xdef\@headdesignation{#1}}

% Page numbering style

\fancypagestyle{plain}{
    \renewcommand{\headrulewidth}{0pt}%
    \fancyhf{}%
    \fancyfoot[R]{\footnotesize\textbf{\thepage}}%    
}

\fancypagestyle{mainStyle}{
	\renewcommand{\headrulewidth}{0pt}%
	\fancyhf{}%
	\fancyfoot[R]{\footnotesize\emph{\leftmark} \hspace{0.25cm} \footnotesize\textbf{\thepage}}%   
}

% START OF THESIS

\newcommand{\thesisBodyStart}{

    % Clear the page
    \cleardoublepage
    
    % Set page numbering back to 1, 2, 3 .. 
    \pagenumbering{arabic} 
}


% TABLE OF CONTENTS, FIGURES AND TABLES
\newcommand{\thesisTOF}{
    \iftotalfigures
        \cleardoublepage
        \listoffigures
        \addcontentsline{toc}{chapter}{List of Figures}
    \fi
}

\newcommand{\thesisTOT}{
    \iftotaltables
        \cleardoublepage
        \listoftables
        \addcontentsline{toc}{chapter}{List of Tables}
    \fi
}
\newcommand{\thesisTOA}{
    \printglossary[title=List of Abbreviations, toctitle=List of Abbreviations]
}

\newcommand{\thesisTables}{
    \cleardoublepage
    \renewcommand{\contentsname}{Table of Contents}
    \setcounter{tocdepth}{4}
    \setcounter{secnumdepth}{4}
    {\setstretch{1.25}
        \tableofcontents
        \thesisTOF
        \thesisTOT
        \thesisTOA
    }
    
    \setlength{\abovedisplayskip}{1pt}
    \setlength{\belowdisplayskip}{15pt}
}


% DECLARATION
\newenvironment{declaration}
    {\cleardoublepage
     \chapter*{\makebox[\linewidth]{Declaration of Originality}}
    }
    {}

% ABSTRACT
\renewenvironment{abstract}
    {\cleardoublepage
        \chapter*{\makebox[\linewidth]{\abstractname}}
        \addcontentsline{toc}{chapter}{Abstract}
    }
    {}

%Keywords
\providecommand{\keywords}[1]
{
  \normalfont	
  \textbf{\textit{Keywords---}} #1
}

% ACKNOWLEDGEMENTS
\newenvironment{acknowledgements}
    {\cleardoublepage
        \chapter*{\makebox[\linewidth]{Acknowledgements}}
        \addcontentsline{toc}{chapter}{Acknowledgements}
    }
    {}


% AFTER TITLE PAGE BUT BEFORE THESIS CONTENT
\newcommand{\afterTitlePage}{

    % Set margins for the rest of the thesis.
    \newgeometry{
        nohead, 
        top=2.5cm,
        bottom=2.5cm,
        right=2.5cm,
        left=4cm,
        marginparwidth=0cm,
        marginparsep=0cm
    }

    % Set spacing and reset numbering
    \pagenumbering{roman}

    % Set double spacing
    \setstretch{1.5}
    \doublehyphendemerits=10000     % No consecutive line hyphens
    \brokenpenalty=10000            % No broken words across columns/pages
    \widowpenalty=9999              % Almost no widows at bottom of page
    \clubpenalty=9999               % Almost no orphans at top of page
    \interfootnotelinepenalty=9999  % Almost never break footnotes
}


% TITLE PAGE

% Print out submission text
\newcommand{\submissionStatement}{
    \setstretch{1.25}
    \@ifundefined{@submissiontext}{Submitted in partial fulfilment of the requirements for \\Degree of \@degree \\ in \@department}{\@submissiontext}
}

% The default two page style title page
\newcommand{\@titlepage}{
    \begin{titlepage}

    % Cover page heading
    \makeatletter
    \centering{\LARGE \@degree \ in \@department \par}\vspace{.5cm}
    
    % the logo
    \includegraphics[height=2in]{\@logo}\par 
    
    % Thesis title, followed by student name and ID
    \vspace{1cm}
    {\Large \bfseries \@title \par}\vspace{1cm}
    {by \par \large \@authorA \par ID: \@IDA \par}
    \ifbool{bauth}{\vspace{1cm}\large \@authorB \par ID: \@IDB \par \vspace{1cm}}{\vspace{5cm}}
    
    % Department, University and Address
    {\large Department of \@department \par}\vspace{0.15cm}
    {\large \@universityname \par}\vspace{0.15cm}
    {\large Chattogram-4349, Bangladesh. \par}
    
    % date
    \vspace{1.5cm}
    {\large \@date \par}
    
    
    % Internal Title page heading for Final Report
    \makeatletter
    \centering{\Large \bfseries \@title \par}
    
    % the logo
    \includegraphics[height=2in]{\@logo}\par 
    
    % submission statement followed by student name, id, supervisor name & designation
    {\large \submissionStatement \par}\vspace{0.5cm}
    
    {by \par \large \@authorA \par ID: \@IDA \par}
    \ifbool{bauth}{\vspace{0.5cm} \large \@authorB \par ID: \@IDB \par \vspace{0.5cm}}{\vspace{3.5cm}}
    
    {Supervised by \par \large \@supervisor \par \@supervisordesignation \par}
    {\large Department of \@department \par}
    
    \vspace{1.5cm}
    {\large \@universityname \par}
    {\large Chattogram-4349, Bangladesh. \par}
    
    \thispagestyle{empty}
    
    
    % % Internal Title page heading for Proposal Report
    % \makeatletter
    % \centering{\large \bfseries \@universityname \par}
    % \centering{\large \bfseries Department of \@department \par}
    % \centering{\large \bfseries Chattogram-4349, Bangladesh. \par}
    % \hrulefill \par
    % \centering{\Large Thesis Proposal \par}
    % \centering{\large Application for the Approval of \@shortdegree \ Engineering Thesis/Project \par}\vspace{1cm}
    
    % \begin{description}[style=multiline,leftmargin=10em,itemsep=2em]
    
    % \item[Student Name \ \ ID ] \ifbool{bauth}{: \@authorA \hspace{7em} Session: \@studentsession \\ : \@IDA \\ : \@authorB \\ : \@IDB}{: \@authorA \hspace{14em} Session : \@studentsession \\ : \@IDA}
    
    % \item[Supervisor Name  Designation  ] \large : \@supervisor \\ : \@supervisordesignation \\ Department of \@department \\ \@universityname
    
    % \item[Department Program] \large : \@department \\ : \@shortdegree \ Engineering
    
    % \item[Tentative Title] \large \bfseries : \nohyphens{\@title}
    
    % \end{description}
    
    % \thispagestyle{empty}
    
	\end{titlepage}
}

\newcommand{\beforeTitlePage}{
    % Set margins for the title pages.
    \newgeometry{
        nohead, 
        top=3cm,
        bottom=3cm,
        right=1.5cm,
        left=3cm,
        marginparwidth=0cm,
        marginparsep=0cm
    }
}


% Renew and recall the maketitle command
\RenewDocumentCommand{\maketitle}{o}{%

  % Call function for before title page
  \beforeTitlePage
  
  %Call function for the title
  \@titlepage
    
  % Call function for after title page       
  \afterTitlePage
}


% Print out supervisors
\newcommand{\printSupervisor}{
    \begin{singlespace}
        \@ifundefined{@supervisor}{}
        {
        \vspace{5em} \rule[0.5em]{20em}{0.5pt} \hspace{5em} \mbox{Chairman}
        
        \@supervisor
        
        \@supervisordesignation
        
        Department of \@department
        
        \@universityname
        }
            
        \@ifundefined{@departmenthead}{}
        {
        \vspace{5em} \rule[0.5em]{20em}{0.5pt} \hspace{5em} Member (Ex-Officio)
        
        \@departmenthead
        
        \@headdesignation \ \& Head
        
        Department of \@department
        
        \@universityname
        }
            
        \@ifundefined{@externalsupervisor}{}
        {
        \vspace{5em} \rule[0.5em]{20em}{0.5pt} \hspace{5em} Member (External)
        
        \@externalsupervisor
        
        \@externaldesignation
        
        \@ifundefined{@externaldepartment}{Department of \@department}{Department of \@externaldepartment}
        
        \@ifundefined{@externaluniversity}{\@universityname}{\@externaluniversity}
        }
    \end{singlespace}
}

% Certification
\newcommand{\certification}
    {\cleardoublepage
    
    The thesis titled \enquote{\bfseries \large \@title} submitted by ID: \@IDA  \ifbool{bauth}{and \@IDB,}{,} Session \@studentsession \ has been accepted as satisfactory in fulfilment of the requirement for the degree of \@degree \ in \@department \ to be awarded by the \@universityname.
    
    \vspace{0.5cm}\makebox[\linewidth]{\Large \bfseries Board of Examiners}
    
    \printSupervisor
    }
    {}


% TITLE / CHAPTERS / FIGURES SPACING

%% Set the chapter title style for the Final report format

\titleformat{\chapter}[display]
{\centering \normalfont\huge\bfseries}{\chaptertitlename\ \thechapter}{0pt}{\Huge}
% Provide spacing
\titlespacing*{\chapter} {0pt}{0pt}{5mm}
% A fake chapter title command to emulate chapter titles
\newcommand{\chapterTitle}[1]{
	\vspace*{-20mm} 
	{\normalfont\huge\bfseries#1 \par} 
	\vspace*{20mm}
}

% % Set the chapter title style for the Proposal report format

% \titleformat{\section}{\normalfont\huge\bfseries}{\thesection \ }{0pt}{\Huge}
% \titlespacing*{\section}{0pt}{0pt}{10pt}

% \renewcommand\thesection{\arabic{section}}
% \renewcommand{\thetable}{\arabic{section}.\arabic{table}}
% \renewcommand{\thefigure}{\arabic{section}.\arabic{figure}}

% \titlecontents{\thesection}
%   [0em]{\addvspace{10pt}\bfseries}
%   {}{}{\titlerule*[1000pc]{.}\contentspage}


% % CUGS Reference
% \newcommand{\CUGS}
%     {\cleardoublepage
    
%     \Large \bfseries CSE Undergraduate Studies (CUGS) Committee Reference : \\
    
%     \large Meeting No : \hspace{50pt} Meeting No : \hspace{50pt} Date : \\
    
%     \par \vspace{2cm}\hspace{240pt}\rule[0pt]{170pt}{1pt}
%     \par \begin{flushright}Signature of the student \end{flushright}
    
%     \par \vspace{2.5cm}\hspace{220pt}\rule[0pt]{190pt}{1pt}
%     \par \begin{flushright}Signature of the Supervisor \end{flushright}
    
%     \par \vspace{2.5cm}\hspace{120pt}\rule[0pt]{290pt}{1pt}
%     \par \begin{flushright}Signature of the Head of the Department \end{flushright}
    
%     \thispagestyle{empty}
    
%     }
%     {}


% REFERENCING
% IEEE style and numeric citation, using bibLatex
\RequirePackage[
    defernumbers=true,
    backend=biber,
    bibencoding=utf8,
    natbib=true,
    hyperref=true,
    backref=true,
    urldate=long,
    style=ieee,
    citestyle=numeric,
    sorting=none
]{biblatex}

% Print out references list
\newcommand{\printReferences}[1][References]{
	
	% break on URL numbers
    \setcounter{biburlnumpenalty}{9000}
    
    % break on URL lower case letters
    \setcounter{biburllcpenalty}{9000}
    
    % break on URL UPPER CASE letters
    \setcounter{biburlucpenalty}{9000}
    
	% Set spacing between items  (was 0.35cm; tightened to keep the reference
	% list within its original page count after adding a citation)
	\setlength\bibitemsep{0.20cm}
	
	% Set the line height to 1.0
	\setstretch{1.0}
	
	% Print out the bibliography with
	\printbibliography[title={References}]
}

% Set the page style to the main style. Set up the chapter
% mark style so that it shows the title of the chapter only.
\pagestyle{mainStyle}
\renewcommand{\chaptermark}[1]{\markboth{{#1}}{}}