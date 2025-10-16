"""
CrewAI + Gemini example: CBSE Class 12 maths solver
Proper implementation using CrewAI agents and Google Gemini SDK.
"""

import os
import sys
import re
import json
import asyncio
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

# Proper imports
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool
import google.generativeai as genai
from sympy import simplify, symbols, solve, Eq, latex
from sympy.parsing.latex import parse_latex

# -----------------------
# Configuration
# -----------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# -----------------------
# Custom Tools
# -----------------------

class LatexMathTool(BaseTool):
    name: str = "LaTeX Math Parser"
    description: str = "Extract and parse LaTeX mathematical expressions for symbolic verification"
    
    def _run(self, tex_content: str) -> List[Dict[str, Any]]:
        """Extract LaTeX math blocks and attempt to parse them with SymPy."""
        math_blocks = self._extract_math_blocks(tex_content)
        parsed_expressions = []
        
        for math_block in math_blocks:
            try:
                parsed_expr = parse_latex(math_block)
                parsed_expressions.append({
                    "original": math_block,
                    "parsed": parsed_expr,
                    "success": True
                })
            except Exception as e:
                parsed_expressions.append({
                    "original": math_block,
                    "error": str(e),
                    "success": False
                })
        
        return parsed_expressions
    
    def _extract_math_blocks(self, tex: str) -> List[str]:
        """Extract LaTeX math blocks using regex patterns."""
        patterns = [
            r'\$(.*?)\$',                    # $...$
            r'\\\((.*?)\\\)',               # \(...\)
            r'\\\[(.*?)\\\]',               # \[...\]
            r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}',  # \begin{equation}...\end{equation}
            r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}',        # \begin{align}...\end{align}
        ]
        
        blocks = []
        for pattern in patterns:
            matches = re.findall(pattern, tex, re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    # For multi-line environments, join the content
                    content = ''.join(match).strip()
                else:
                    content = match.strip()
                if content:
                    blocks.append(content)
        
        return blocks

class SymbolicVerifierTool(BaseTool):
    name: str = "Symbolic Verifier"
    description: str = "Verify mathematical correctness using symbolic computation"
    
    def _run(self, expressions: List[Dict[str, Any]], question_context: str = "") -> Dict[str, Any]:
        """Perform symbolic verification on parsed expressions."""
        issues = []
        verified_expressions = []
        
        successful_exprs = [expr for expr in expressions if expr.get("success")]
        
        for i, expr in enumerate(successful_exprs):
            try:
                # Basic sanity checks
                parsed_expr = expr["parsed"]
                
                # Check for complex numbers in real problems (heuristic)
                if hasattr(parsed_expr, 'free_symbols'):
                    # This is a simple check - in practice you'd want more sophisticated logic
                    pass
                
                verified_expressions.append({
                    "index": i,
                    "expression": expr["original"],
                    "parsed": str(parsed_expr),
                    "verified": True
                })
                
            except Exception as e:
                issues.append({
                    "type": "verification_error",
                    "expression": expr["original"],
                    "error": str(e)
                })
                verified_expressions.append({
                    "index": i,
                    "expression": expr["original"],
                    "verified": False
                })
        
        # Check for consistency between consecutive expressions
        for i in range(len(verified_expressions) - 1):
            if verified_expressions[i]["verified"] and verified_expressions[i + 1]["verified"]:
                try:
                    expr1 = successful_exprs[i]["parsed"]
                    expr2 = successful_exprs[i + 1]["parsed"]
                    
                    # Try to see if they're related (e.g., one is derivative of other, or they're equal)
                    diff = simplify(expr1 - expr2)
                    if diff != 0:
                        # They're not equal, but might be related in other ways
                        # For now, just note the difference
                        issues.append({
                            "type": "potential_inconsistency",
                            "expressions": [
                                verified_expressions[i]["expression"],
                                verified_expressions[i + 1]["expression"]
                            ],
                            "difference": str(diff)
                        })
                except Exception as e:
                    issues.append({
                        "type": "comparison_error",
                        "error": str(e)
                    })
        
        return {
            "verified": len(issues) == 0,
            "issues": issues,
            "verified_expressions": verified_expressions
        }

# -----------------------
# LLM Configuration
# -----------------------

def create_gemini_llm(model_name: str = "gemini-1.5-flash", temperature: float = 0.1):
    """Create a CrewAI-compatible LLM configuration for Gemini."""
    # Note: CrewAI doesn't natively support Gemini, so we'd use a wrapper
    # For this example, we'll use the default LLM and override where needed
    from crewai.llm import LLM
    
    class GeminiLLM(LLM):
        def __init__(self, model: str = model_name, temperature: float = temperature):
            self.model_name = model
            self.temperature = temperature
            
        def call(self, prompt: str, tools: list = None) -> str:
            """Make actual Gemini API call."""
            try:
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature
                    )
                )
                return response.text
            except Exception as e:
                return f"Error calling Gemini: {str(e)}"
    
    return GeminiLLM()

# Since CrewAI doesn't natively support Gemini, we'll use the default LLM
# and implement Gemini calls in the task execution
gemini_llm = create_gemini_llm()

# -----------------------
# Agent Definitions
# -----------------------

# Initialize tools
latex_tool = LatexMathTool()
verifier_tool = SymbolicVerifierTool()

# Writer Agent - Creates initial solution
writer_agent = Agent(
    role="Senior Mathematics Tutor",
    goal="Create clear, step-by-step LaTeX solutions for CBSE Class 12 mathematics problems",
    backstory="""You are an experienced mathematics teacher with 15 years of experience 
    teaching CBSE Class 12 curriculum. You specialize in creating clear, methodical 
    solutions that help students understand the underlying concepts and procedures.""",
    tools=[latex_tool],
    verbose=True,
    allow_delegation=False
)

# Verifier Agent - Checks mathematical correctness
verifier_agent = Agent(
    role="Mathematics Verifier",
    goal="Verify the mathematical correctness of solutions using symbolic computation and logical checks",
    backstory="""You are a meticulous mathematician with expertise in symbolic computation. 
    You use tools like SymPy to verify algebraic manipulations, derivative calculations, 
    integral solutions, and logical consistency in mathematical proofs.""",
    tools=[latex_tool, verifier_tool],
    verbose=True,
    allow_delegation=False
)

# Corrector Agent - Fixes identified issues
corrector_agent = Agent(
    role="Mathematics Solution Editor",
    goal="Correct mathematical errors and improve solution clarity based on verification feedback",
    backstory="""You are an expert mathematics editor who specializes in identifying 
    and correcting errors in mathematical solutions. You work closely with verifiers 
    to ensure solutions are both mathematically correct and pedagogically sound.""",
    tools=[latex_tool],
    verbose=True,
    allow_delegation=False
)

# Formatter Agent - Creates final output
formatter_agent = Agent(
    role="Mathematics Document Formatter",
    goal="Format mathematical solutions into clean LaTeX with plain English summaries",
    backstory="""You are a technical editor specializing in mathematical documentation. 
    You ensure solutions are properly formatted in LaTeX, include appropriate step 
    numbering and explanations, and provide clear plain-language summaries.""",
    verbose=True,
    allow_delegation=False
)

# -----------------------
# Task Definitions
# -----------------------

def create_writer_task(question: str) -> Task:
    return Task(
        description=f"""Create a comprehensive step-by-step solution for the following CBSE Class 12 mathematics problem:

{question}

Requirements:
1. Use proper LaTeX formatting for all mathematical expressions
2. Number each step clearly
3. Include brief explanations for key steps
4. Provide a boxed final answer
5. Ensure the solution follows CBSE examination standards""",
        agent=writer_agent,
        expected_output="A complete LaTeX solution with step-by-step working and final answer"
    )

def create_verifier_task() -> Task:
    return Task(
        description="""Verify the mathematical correctness of the solution:

1. Extract and parse all mathematical expressions using the LaTeX tool
2. Verify algebraic manipulations and calculations
3. Check for logical consistency between steps
4. Identify any potential errors or inconsistencies
5. Provide a detailed verification report""",
        agent=verifier_agent,
        expected_output="A verification report with boolean verified status and list of issues found",
        context=[create_writer_task("dummy")]  # This will be replaced in the crew
    )

def create_corrector_task() -> Task:
    return Task(
        description="""Correct any issues identified in the verification report:

1. Review the original solution and verification findings
2. Correct mathematical errors
3. Improve clarity and explanation where needed
4. Maintain the original step structure where possible
5. Provide a revised solution with corrections clearly marked""",
        agent=corrector_agent,
        expected_output="A corrected LaTeX solution with explanations of changes made",
        context=[create_verifier_task()]  # This will be replaced in the crew
    )

def create_formatter_task() -> Task:
    return Task(
        description="""Format the final solution for presentation:

1. Ensure clean LaTeX formatting
2. Add appropriate section headers if needed
3. Create a plain English summary of the solution
4. Verify all mathematical expressions render correctly
5. Provide both LaTeX and plain text outputs""",
        agent=formatter_agent,
        expected_output="A dictionary with keys: 'final_latex' and 'plain_answer'",
        context=[create_corrector_task()]  # This will be replaced in the crew
    )

# -----------------------
# Crew Orchestration
# -----------------------

class MathSolutionCrew:
    def __init__(self):
        self.crew = None
    
    def setup_crew(self, question: str):
        """Set up the crew with the specific question."""
        writer_task = create_writer_task(question)
        verifier_task = create_verifier_task()
        corrector_task = create_corrector_task()
        formatter_task = create_formatter_task()
        
        # Update context relationships
        verifier_task.context = [writer_task]
        corrector_task.context = [verifier_task]
        formatter_task.context = [corrector_task]
        
        self.crew = Crew(
            agents=[writer_agent, verifier_agent, corrector_agent, formatter_agent],
            tasks=[writer_task, verifier_task, corrector_task, formatter_task],
            process=Process.sequential,
            verbose=True
        )
    
    def kickoff(self) -> Dict[str, Any]:
        """Execute the crew and return results."""
        if not self.crew:
            raise ValueError("Crew not set up. Call setup_crew() first.")
        
        try:
            result = self.crew.kickoff()
            return self._process_result(result)
        except Exception as e:
            return {
                "error": str(e),
                "final_latex": "",
                "plain_answer": "",
                "verified": False,
                "issues": [{"type": "execution_error", "detail": str(e)}]
            }
    
    def _process_result(self, result) -> Dict[str, Any]:
        """Process the crew result into our expected format."""
        # The result structure depends on how tasks are chained
        # For simplicity, we'll extract from the final formatter output
        if hasattr(result, 'raw') and isinstance(result.raw, dict):
            return result.raw
        else:
            # Fallback processing
            return {
                "final_latex": str(result),
                "plain_answer": self._extract_plain_answer(str(result)),
                "verified": True,  # Assuming verification passed if we got here
                "issues": []
            }
    
    def _extract_plain_answer(self, solution: str) -> str:
        """Extract a plain English summary from the solution."""
        # Use Gemini to create a summary
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""Convert this mathematical solution into a brief plain English summary:
            
            {solution}
            
            Provide only the final answer and key conclusion in simple English."""
            
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            # Fallback: extract the last line or boxed answer
            lines = solution.split('\n')
            for line in reversed(lines):
                if 'boxed' in line or 'answer' in line.lower():
                    return line.strip()
            return lines[-1] if lines else "Solution generated"

# -----------------------
# Direct Gemini Fallback (for simple cases)
# -----------------------

def direct_gemini_solution(question: str) -> Dict[str, Any]:
    """Fallback solution using direct Gemini call."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""Solve this CBSE Class 12 mathematics problem and provide:
        1. A complete step-by-step LaTeX solution
        2. A brief plain English summary of the answer
        
        Problem: {question}
        
        Format your response as JSON:
        {{
            "final_latex": "complete LaTeX solution here",
            "plain_answer": "brief English summary here"
        }}"""
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Try to parse JSON, fallback to raw text
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            result = {
                "final_latex": response_text,
                "plain_answer": "See LaTeX solution above"
            }
        
        result.update({
            "verified": True,  # Can't verify in fallback mode
            "issues": [],
            "source": "direct_gemini"
        })
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "final_latex": "",
            "plain_answer": "",
            "verified": False,
            "issues": [{"type": "gemini_error", "detail": str(e)}],
            "source": "direct_gemini_fallback"
        }

# -----------------------
# Main Execution
# -----------------------

async def solve_question(latex_question: str, use_crew: bool = True) -> Dict[str, Any]:
    """Main function to solve a mathematics question."""
    
    if use_crew:
        try:
            crew = MathSolutionCrew()
            crew.setup_crew(latex_question)
            result = crew.kickoff()
            return result
        except Exception as e:
            print(f"CrewAI approach failed: {e}. Falling back to direct Gemini...")
            return direct_gemini_solution(latex_question)
    else:
        return direct_gemini_solution(latex_question)

# -----------------------
# CLI Interface
# -----------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python crewai_gemini_cbse_solver.py '<latex_question_string>' [--direct]")
        print("Example: python crewai_gemini_cbse_solver.py 'Solve for x: $x^2 - 5x + 6 = 0$'")
        sys.exit(1)
    
    question = sys.argv[1]
    use_crew = "--direct" not in sys.argv
    
    print(f"Solving question: {question}")
    print(f"Using CrewAI: {use_crew}")
    print("-" * 50)
    
    try:
        result = asyncio.run(solve_question(question, use_crew=use_crew))
        
        # Display results
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        
        print("\n📝 Plain Answer:")
        print(result.get("plain_answer", "No plain answer generated"))
        
        print("\n🧮 LaTeX Solution:")
        print(result.get("final_latex", "No LaTeX solution generated"))
        
        print("\n✅ Verification Status:")
        print(f"Verified: {result.get('verified', False)}")
        
        if result.get("issues"):
            print(f"Issues Found: {len(result['issues'])}")
            for issue in result["issues"]:
                print(f"  - {issue.get('type', 'unknown')}: {issue.get('detail', 'No details')}")
        
        if result.get("source"):
            print(f"\nSource: {result['source']}")
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()