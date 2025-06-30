#!/usr/bin/env python3
"""
TTS Streaming Client with Time-to-First-Byte Measurement

This client calls the TTS endpoint and measures the time to first byte received
from the streaming audio response, excluding the WAV header.
"""

import requests
import time
import argparse
import sys
import struct
from pathlib import Path


# Available voices in the Orpheus TTS model
AVAILABLE_VOICES = ["zoe", "zac", "jess", "leo", "mia", "julia", "leah", "tara"]


def parse_wav_header(data):
    """
    Parse WAV header to determine header size.
    Returns the size of the WAV header in bytes.
    """
    if len(data) < 44:  # Minimum WAV header size
        return 0
    
    try:
        # Check for RIFF signature
        if data[:4] != b'RIFF':
            return 0
        
        # Check for WAVE format
        if data[8:12] != b'WAVE':
            return 0
        
        # Find the data chunk
        pos = 12
        while pos < len(data) - 8:
            chunk_id = data[pos:pos+4]
            chunk_size = struct.unpack('<I', data[pos+4:pos+8])[0]
            
            if chunk_id == b'data':
                return pos + 8  # Position after data chunk header
            
            pos += 8 + chunk_size
            
        # If we can't find data chunk, assume standard 44-byte header
        return 44
    except:
        # If parsing fails, assume standard 44-byte header
        return 44


def measure_tts_streaming(base_url, prompt, voice, output_file="output.wav", timeout=30):
    """
    Call the TTS endpoint and measure time to first audio byte (excluding WAV header).
    
    Args:
        base_url (str): Base URL of the TTS server (e.g., "http://localhost:8080")
        prompt (str): Text prompt to convert to speech
        voice (str): Voice to use for TTS
        output_file (str): Output filename for the audio file
        timeout (int): Request timeout in seconds
    
    Returns:
        dict: Dictionary containing timing metrics and response info
    """
    url = f"{base_url}/tts"
    params = {"prompt": prompt, "voice": voice}
    
    print(f"Making TTS request to: {url}")
    print(f"Voice: {voice}")
    print(f"Prompt: {prompt}")
    print(f"Output file: {output_file}")
    print("-" * 50)
    
    # Prepare timing variables
    request_start_time = None
    first_byte_time = None
    first_audio_byte_time = None
    total_bytes_received = 0
    header_bytes_received = 0
    audio_bytes_received = 0
    header_parsed = False
    header_size = 0
    accumulated_data = b''
    
    try:
        # Start timing the request
        request_start_time = time.time()
        
        # Make the streaming request
        with requests.get(url, params=params, stream=True, timeout=timeout) as response:
            # Check if request was successful
            response.raise_for_status()
            
            print(f"Response status: {response.status_code}")
            print(f"Content-Type: {response.headers.get('Content-Type', 'Not specified')}")
            
            # Open output file for writing binary data
            with open(output_file, 'wb') as f:
                # Iterate through the streaming response
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # Filter out keep-alive chunks
                        # Record time of first byte if not already recorded
                        if first_byte_time is None:
                            first_byte_time = time.time()
                            time_to_first_byte = first_byte_time - request_start_time
                            print(f"✓ First byte received after: {time_to_first_byte:.4f} seconds")
                        
                        # Write chunk to file and update byte counter
                        f.write(chunk)
                        total_bytes_received += len(chunk)
                        
                        # Handle WAV header parsing and audio byte detection
                        if not header_parsed:
                            accumulated_data += chunk
                            
                            # Try to parse header once we have enough data
                            if len(accumulated_data) >= 44:  # Minimum WAV header size
                                header_size = parse_wav_header(accumulated_data)
                                header_parsed = True
                                
                                print(f"  WAV header detected: {header_size} bytes")
                                
                                # Check if we already have audio data beyond the header
                                if len(accumulated_data) > header_size:
                                    first_audio_byte_time = first_byte_time  # Approximate, since we got it in same chunk
                                    audio_bytes_in_first_chunks = len(accumulated_data) - header_size
                                    audio_bytes_received += audio_bytes_in_first_chunks
                                    header_bytes_received = header_size
                                    
                                    time_to_first_audio_byte = first_audio_byte_time - request_start_time
                                    print(f"✓ First AUDIO byte received after: {time_to_first_audio_byte:.4f} seconds")
                        else:
                            # We're past the header, this is all audio data
                            if first_audio_byte_time is None:
                                first_audio_byte_time = time.time()
                                time_to_first_audio_byte = first_audio_byte_time - request_start_time
                                print(f"✓ First AUDIO byte received after: {time_to_first_audio_byte:.4f} seconds")
                            
                            audio_bytes_received += len(chunk)
                        
                        # Show progress (optional, can be commented out for cleaner output)
                        if total_bytes_received % (1024 * 10) == 0:  # Every 10KB
                            print(f"  Received: {total_bytes_received:,} bytes (Audio: {audio_bytes_received:,})", end='\r')
            
            # Final timing calculations
            request_end_time = time.time()
            total_request_time = request_end_time - request_start_time
            
            # Clear the progress line
            print(" " * 70, end='\r')
            
            print(f"✓ Audio stream completed successfully!")
            print(f"  Total bytes received: {total_bytes_received:,} bytes")
            print(f"  WAV header bytes: {header_bytes_received:,} bytes")
            print(f"  Audio data bytes: {audio_bytes_received:,} bytes")
            print(f"  Total request time: {total_request_time:.4f} seconds")
            
            if first_byte_time:
                time_to_first_byte = first_byte_time - request_start_time
                print(f"  Time to first byte (any): {time_to_first_byte:.4f} seconds")
                
            if first_audio_byte_time:
                time_to_first_audio_byte = first_audio_byte_time - request_start_time
                print(f"  Time to first AUDIO byte: {time_to_first_audio_byte:.4f} seconds")
                
                streaming_time = request_end_time - first_audio_byte_time
                print(f"  Audio streaming time: {streaming_time:.4f} seconds")
                
                if streaming_time > 0 and audio_bytes_received > 0:
                    throughput = audio_bytes_received / streaming_time
                    print(f"  Audio throughput: {throughput:.2f} bytes/second")
            
            return {
                "success": True,
                "voice": voice,
                "status_code": response.status_code,
                "total_bytes": total_bytes_received,
                "header_bytes": header_bytes_received,
                "audio_bytes": audio_bytes_received,
                "time_to_first_byte": time_to_first_byte if first_byte_time else None,
                "time_to_first_audio_byte": time_to_first_audio_byte if first_audio_byte_time else None,
                "total_request_time": total_request_time,
                "output_file": output_file,
                "prompt": prompt
            }
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Request failed: {e}"
        print(f"✗ {error_msg}")
        return {
            "success": False,
            "voice": voice,
            "error": error_msg,
            "prompt": prompt
        }
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"✗ {error_msg}")
        return {
            "success": False,
            "voice": voice,
            "error": error_msg,
            "prompt": prompt
        }


def test_all_voices(base_url, prompt, output_prefix="tts_output", timeout=30):
    """
    Test all available voices with the same prompt and measure TTFB for each.
    
    Args:
        base_url (str): Base URL of the TTS server
        prompt (str): Text prompt to convert to speech
        output_prefix (str): Prefix for output files (voice name will be appended)
        timeout (int): Request timeout in seconds
    
    Returns:
        list: List of result dictionaries for each voice
    """
    results = []
    
    print("=" * 80)
    print("TTS Multi-Voice Streaming Test - Audio Time to First Byte Measurement")
    print("=" * 80)
    print(f"Testing {len(AVAILABLE_VOICES)} voices: {', '.join(AVAILABLE_VOICES)}")
    print(f"Prompt: {prompt}")
    print("=" * 80)
    
    for i, voice in enumerate(AVAILABLE_VOICES, 1):
        print(f"\n[{i}/{len(AVAILABLE_VOICES)}] Testing voice: {voice}")
        print("=" * 60)
        
        # Generate output filename
        output_file = f"{output_prefix}_{voice}.wav"
        
        # Test this voice
        result = measure_tts_streaming(
            base_url=base_url,
            prompt=prompt,
            voice=voice,
            output_file=output_file,
            timeout=timeout
        )
        
        results.append(result)
        
        # Print summary for this voice
        if result["success"]:
            print(f"✓ {voice}: TTFB = {result.get('time_to_first_audio_byte', 'N/A'):.4f}s, "
                  f"Audio bytes = {result.get('audio_bytes', 0):,}")
        else:
            print(f"✗ {voice}: Failed - {result.get('error', 'Unknown error')}")
        
        # Add a small delay between requests to be respectful to the server
        if i < len(AVAILABLE_VOICES):
            time.sleep(0.5)
    
    return results


def print_summary_report(results):
    """Print a summary report of all voice tests."""
    print("\n" + "=" * 80)
    print("SUMMARY REPORT - All Voices TTFB Comparison")
    print("=" * 80)
    
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    if successful_results:
        print(f"✓ Successful: {len(successful_results)}/{len(results)} voices")
        print("\nTTFB Rankings (fastest to slowest):")
        print("-" * 60)
        
        # Sort by TTFB for audio bytes
        sorted_results = sorted(
            successful_results, 
            key=lambda x: x.get('time_to_first_audio_byte', float('inf'))
        )
        
        for i, result in enumerate(sorted_results, 1):
            ttfb = result.get('time_to_first_audio_byte', 0)
            audio_bytes = result.get('audio_bytes', 0)
            total_time = result.get('total_request_time', 0)
            voice = result['voice']
            
            print(f"{i:2d}. {voice:8s}: {ttfb:.4f}s TTFB | "
                  f"{audio_bytes:6,} bytes | {total_time:.4f}s total")
        
        # Calculate statistics
        ttfb_times = [r.get('time_to_first_audio_byte', 0) for r in successful_results]
        if ttfb_times:
            avg_ttfb = sum(ttfb_times) / len(ttfb_times)
            min_ttfb = min(ttfb_times)
            max_ttfb = max(ttfb_times)
            
            print(f"\nStatistics:")
            print(f"  Average TTFB: {avg_ttfb:.4f}s")
            print(f"  Fastest TTFB: {min_ttfb:.4f}s ({sorted_results[0]['voice']})")
            print(f"  Slowest TTFB: {max_ttfb:.4f}s ({sorted_results[-1]['voice']})")
            print(f"  TTFB Range:   {max_ttfb - min_ttfb:.4f}s")
    
    if failed_results:
        print(f"\n✗ Failed: {len(failed_results)} voices")
        for result in failed_results:
            print(f"  - {result['voice']}: {result.get('error', 'Unknown error')}")
    
    print("\nGenerated files:")
    for result in successful_results:
        print(f"  - {result['output_file']}")


def main():
    parser = argparse.ArgumentParser(description="TTS Multi-Voice Streaming Client with Audio TTFB Measurement")
    parser.add_argument(
        "--url", 
        default="http://localhost:8080", 
        help="Base URL of the TTS server (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--prompt", 
        default="Hey What's up? <laugh> What have you been doing today? So happy to see you!",
        help="Text prompt to convert to speech"
    )
    parser.add_argument(
        "--output-prefix", 
        default="tts_output",
        help="Prefix for output audio files (default: tts_output)"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=30,
        help="Request timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--voice",
        choices=AVAILABLE_VOICES,
        help=f"Test only a specific voice. Available: {', '.join(AVAILABLE_VOICES)}"
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voices and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_voices:
        print("Available voices:")
        for voice in AVAILABLE_VOICES:
            print(f"  - {voice}")
        return
    
    if args.voice:
        # Test single voice
        print("=" * 60)
        print("TTS Single Voice Streaming Client - Audio Time to First Byte Measurement")
        print("=" * 60)
        
        output_file = f"{args.output_prefix}_{args.voice}.wav"
        result = measure_tts_streaming(
            base_url=args.url,
            prompt=args.prompt,
            voice=args.voice,
            output_file=output_file,
            timeout=args.timeout
        )
        
        print("-" * 50)
        if result["success"]:
            print("✓ Test completed successfully!")
            print(f"Audio saved to: {result['output_file']}")
            
            if result.get("time_to_first_audio_byte"):
                print(f"\n📊 Key Metrics:")
                print(f"   Voice: {result['voice']}")
                print(f"   Time to First Byte (any): {result.get('time_to_first_byte', 'N/A'):.4f} seconds")
                print(f"   Time to First AUDIO Byte: {result['time_to_first_audio_byte']:.4f} seconds")
                print(f"   Audio bytes received: {result.get('audio_bytes', 0):,}")
            elif result.get("time_to_first_byte"):
                print(f"\n📊 Key Metric:")
                print(f"   Time to First Byte: {result['time_to_first_byte']:.4f} seconds")
        else:
            print("✗ Test failed!")
            sys.exit(1)
    else:
        # Test all voices
        results = test_all_voices(
            base_url=args.url,
            prompt=args.prompt,
            output_prefix=args.output_prefix,
            timeout=args.timeout
        )
        
        # Print summary report
        print_summary_report(results)


if __name__ == "__main__":
    main() 