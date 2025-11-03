from typing import List, Dict, Any, Optional, Iterator
import logging
from .clients.llm_client import LLMClient
from .prompt import PromptLoader
from .frame import Frame
from .audio_processor import AudioTranscript

logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self, client: LLMClient, model: str, prompt_loader: PromptLoader, temperature: float, user_prompt: str = ""):
        self.client = client
        self.model = model
        self.prompt_loader = prompt_loader
        self.temperature = temperature
        self.user_prompt = user_prompt
        self.context_length: int = 4096
        self._load_prompts()
        self.previous_analyses = []
        
    def _format_user_prompt(self) -> str:
        if self.user_prompt:
            return f"I want to know {self.user_prompt}"
        return ""
        
    def _load_prompts(self):
        self.frame_prompt = self.prompt_loader.get_by_index(0)
        self.video_prompt = self.prompt_loader.get_by_index(1)

    def _format_previous_analyses(self) -> str:
        if not self.previous_analyses:
            return ""
        formatted_analyses = [f"Frame {i}\n{analysis.get('response', 'No analysis available')}\n" for i, analysis in enumerate(self.previous_analyses)]
        return "\n".join(formatted_analyses)

    def analyze_frame(self, frame: Frame, stream: bool = False) -> Iterator[str] | Dict[str, Any]:
        """Analyze a single frame. Can return a stream of text chunks or a final dictionary."""
        prompt = self.frame_prompt.replace("{PREVIOUS_FRAMES}", self._format_previous_analyses())
        prompt = prompt.replace("{prompt}", self._format_user_prompt())
        prompt = f"{prompt}\nThis is frame {frame.number} captured at {frame.timestamp:.2f} seconds."
        
        try:
            response_generator = self.client.generate(
                prompt=prompt,
                image_path=str(frame.path),
                model=self.model,
                temperature=self.temperature,
                num_predict=300,
                context_length=self.context_length,
                stream=stream
            )
            
            if stream:
                return response_generator
            else:
                # Consume the generator to get the full response
                full_response = "".join(list(response_generator))
                analysis_result = {"response": full_response}
                self.previous_analyses.append(analysis_result)
                return analysis_result

        except Exception as e:
            logger.error(f"Error analyzing frame {frame.number}: {e}")
            error_result = {"response": f"Error analyzing frame {frame.number}: {str(e)}"}
            if not stream:
                self.previous_analyses.append(error_result)
            
            if stream:
                yield f"Error: {e}"
            else:
                return error_result

    def reconstruct_video(self, frame_analyses: List[Dict[str, Any]], frames: List[Frame], 
                         transcript: Optional[AudioTranscript] = None) -> Dict[str, Any]:
        frame_notes = []
        for i, (frame, analysis) in enumerate(zip(frames, frame_analyses)):
            frame_note = (
                f"Frame {i} ({frame.timestamp:.2f}s):\n"
                f"{analysis.get('response', 'No analysis available')}"
            )
            frame_notes.append(frame_note)
        
        analysis_text = "\n\n".join(frame_notes)
        
        first_frame_text = frame_analyses[0].get('response', '') if frame_analyses else ''
        
        transcript_text = transcript.text if transcript and transcript.text.strip() else ""
        
        prompt = self.video_prompt.replace("{prompt}", self._format_user_prompt())
        prompt = prompt.replace("{FRAME_NOTES}", analysis_text)
        prompt = prompt.replace("{FIRST_FRAME}", first_frame_text)
        prompt = prompt.replace("{TRANSCRIPT}", transcript_text)
        
        try:
            # Final reconstruction is not streamed
            response_generator = self.client.generate(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                num_predict=1000,
                context_length=self.context_length,
                stream=False
            )
            full_response = "".join(list(response_generator))
            logger.info("Successfully reconstructed video description")
            return {"response": full_response}
        except Exception as e:
            logger.error(f"Error reconstructing video: {e}")
            return {"response": f"Error reconstructing video: {str(e)}"}