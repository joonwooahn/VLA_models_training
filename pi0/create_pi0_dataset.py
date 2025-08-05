import os
import pandas as pd
from pathlib import Path
import numpy as np
import logging
import shutil
import argparse
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 중복 코드 제거 - run_ablation_study.py에서 인덱스를 직접 받음

def transform_state_action(df, state_indices, action_indices, video_mode="robotview", robot_type="allex"):
    """Transform state and action based on provided indices"""
    # observation.state 변환
    df['observation.state'] = df['observation.state'].apply(
        lambda x: [np.array(x)[i] for i in state_indices] if isinstance(x, (list, np.ndarray)) else x
    )
    
    # action 변환
    df['action'] = df['action'].apply(
        lambda x: [np.array(x)[i] for i in action_indices] if isinstance(x, (list, np.ndarray)) else x
    )
    
    # video_mode에 따른 이미지 필드 처리
    if video_mode == "robotview":
        # robotview만 사용하는 경우 추가 카메라들 제거
        additional_cameras = []
        
        # 로봇 타입에 따른 메인 카메라 키 결정
        if robot_type == "franka":
            main_camera_key = "observation.images.agentview"
        else:
            main_camera_key = "observation.images.robot0_robotview"
        
        # 메인 카메라를 제외한 모든 이미지 컬럼 찾기
        for col in df.columns:
            if 'observation.images.' in col and col != main_camera_key:
                additional_cameras.append(col)
        
        if additional_cameras:
            df = df.drop(columns=additional_cameras)
            logger.info(f"Removed additional cameras from dataframe: {additional_cameras}")
    # multiview인 경우 모든 이미지 필드 유지
    
    return df

def generate_episodes_stats(data_dir, meta_dir, source_meta_dir, video_mode="robotview", robot_type="allex"):
    """원본 episodes_stats.jsonl을 복사하고 변환된 데이터에 맞게 stats 필드만 업데이트합니다."""
    import json
    
    # 원본 episodes_stats.jsonl 읽기
    source_stats_file = source_meta_dir / "episodes_stats.jsonl"
    if not source_stats_file.exists():
        logger.error(f"Source episodes_stats.jsonl not found: {source_stats_file}")
        return
    
    # 원본 통계 정보 읽기
    original_stats = []
    with open(source_stats_file, 'r') as f:
        for line in f:
            original_stats.append(json.loads(line.strip()))
    
    # 변환된 데이터에서 통계 계산
    transformed_stats = {}
    for parquet_file in data_dir.glob("**/*.parquet"):
        try:
            df = pd.read_parquet(parquet_file)
            episode_index = int(df['episode_index'].iloc[0]) if 'episode_index' in df.columns else 0
            
            # 변환된 데이터의 모든 필드에 대해 통계 계산
            stats_dict = {}
            
            # observation.state (60차원으로 변환됨)
            state_data = np.array([np.array(x) for x in df['observation.state']])
            stats_dict["observation.state"] = {
                "min": state_data.min(axis=0).tolist(),
                "max": state_data.max(axis=0).tolist(),
                "mean": state_data.mean(axis=0).tolist(),
                "std": state_data.std(axis=0).tolist(),
                "count": [len(df)]
            }
            
            # action (21차원으로 변환됨)
            action_data = np.array([np.array(x) for x in df['action']])
            stats_dict["action"] = {
                "min": action_data.min(axis=0).tolist(),
                "max": action_data.max(axis=0).tolist(),
                "mean": action_data.mean(axis=0).tolist(),
                "std": action_data.std(axis=0).tolist(),
                "count": [len(df)]
            }
            
            # 로봇 타입에 따른 메인 카메라 키 결정
            if robot_type == "franka":
                main_camera_key = "observation.images.agentview"
            else:
                main_camera_key = "observation.images.robot0_robotview"
            
            # observation.images (원본에서 복사됨)
            if main_camera_key in df.columns:
                camera_data = np.array([np.array(x) for x in df[main_camera_key]])
                stats_dict[main_camera_key] = {
                    "min": camera_data.min(axis=0).tolist(),
                    "max": camera_data.max(axis=0).tolist(),
                    "mean": camera_data.mean(axis=0).tolist(),
                    "std": camera_data.std(axis=0).tolist(),
                    "count": [len(df)]
                }
            
            # multiview 모드인 경우 추가 카메라들의 통계도 포함
            additional_cameras = []
            # 메인 카메라를 제외한 모든 이미지 컬럼 찾기
            for col in df.columns:
                if 'observation.images.' in col and col != main_camera_key:
                    additional_cameras.append(col)
            
            for camera in additional_cameras:
                if video_mode == "multiview" and camera in df.columns:
                    camera_data = np.array([np.array(x) for x in df[camera]])
                    stats_dict[camera] = {
                        "min": camera_data.min(axis=0).tolist(),
                        "max": camera_data.max(axis=0).tolist(),
                        "mean": camera_data.mean(axis=0).tolist(),
                        "std": camera_data.std(axis=0).tolist(),
                        "count": [len(df)]
                    }
            
            # next.done (boolean)
            if 'next.done' in df.columns:
                done_data = np.array(df['next.done'])
                stats_dict["next.done"] = {
                    "min": [bool(done_data.min())],
                    "max": [bool(done_data.max())],
                    "mean": [float(done_data.mean())],
                    "std": [float(done_data.std())],
                    "count": [len(df)]
                }
            
            # next.success (boolean)
            if 'next.success' in df.columns:
                success_data = np.array(df['next.success'])
                stats_dict["next.success"] = {
                    "min": [bool(success_data.min())],
                    "max": [bool(success_data.max())],
                    "mean": [float(success_data.mean())],
                    "std": [float(success_data.std())],
                    "count": [len(df)]
                }
            
            # next.reward (float)
            if 'next.reward' in df.columns:
                reward_data = np.array(df['next.reward'])
                stats_dict["next.reward"] = {
                    "min": [float(reward_data.min())],
                    "max": [float(reward_data.max())],
                    "mean": [float(reward_data.mean())],
                    "std": [float(reward_data.std())],
                    "count": [len(df)]
                }
            
            # timestamp (float)
            if 'timestamp' in df.columns:
                timestamp_data = np.array(df['timestamp'])
                stats_dict["timestamp"] = {
                    "min": [float(timestamp_data.min())],
                    "max": [float(timestamp_data.max())],
                    "mean": [float(timestamp_data.mean())],
                    "std": [float(timestamp_data.std())],
                    "count": [len(df)]
                }
            
            # frame_index (int)
            if 'frame_index' in df.columns:
                frame_data = np.array(df['frame_index'])
                stats_dict["frame_index"] = {
                    "min": [int(frame_data.min())],
                    "max": [int(frame_data.max())],
                    "mean": [float(frame_data.mean())],
                    "std": [float(frame_data.std())],
                    "count": [len(df)]
                }
            
            # episode_index (int)
            if 'episode_index' in df.columns:
                episode_data = np.array(df['episode_index'])
                stats_dict["episode_index"] = {
                    "min": [int(episode_data.min())],
                    "max": [int(episode_data.max())],
                    "mean": [float(episode_data.mean())],
                    "std": [float(episode_data.std())],
                    "count": [len(df)]
                }
            
            # index (int)
            if 'index' in df.columns:
                index_data = np.array(df['index'])
                stats_dict["index"] = {
                    "min": [int(index_data.min())],
                    "max": [int(index_data.max())],
                    "mean": [float(index_data.mean())],
                    "std": [float(index_data.std())],
                    "count": [len(df)]
                }
            
            # task_index (int)
            if 'task_index' in df.columns:
                task_data = np.array(df['task_index'])
                stats_dict["task_index"] = {
                    "min": [int(task_data.min())],
                    "max": [int(task_data.max())],
                    "mean": [float(task_data.mean())],
                    "std": [float(task_data.std())],
                    "count": [len(df)]
                }
            
            transformed_stats[episode_index] = stats_dict
        except Exception as e:
            logger.error(f"Error processing stats for {parquet_file}: {e}")
    
    # 원본 통계를 기반으로 새로운 episodes_stats.jsonl 생성
    updated_stats = []
    for original_stat in original_stats:
        episode_index = original_stat["episode_index"]
        if episode_index in transformed_stats:
            # 원본 통계를 복사하고 변환된 데이터의 통계로 업데이트
            updated_stat = original_stat.copy()
            updated_stats_dict = transformed_stats[episode_index]
            
            # 이미지 필드는 원본에서 그대로 복사
            if "stats" in original_stat:
                # 로봇 타입에 따른 메인 카메라 키 결정
                if robot_type == "franka":
                    main_camera_key = "observation.images.agentview"
                else:
                    main_camera_key = "observation.images.robot0_robotview"
                
                # 메인 카메라는 항상 복사
                if main_camera_key in original_stat["stats"]:
                    updated_stats_dict[main_camera_key] = original_stat["stats"][main_camera_key]
                
                # multiview 모드인 경우 추가 카메라들도 복사
                additional_cameras = []
                for key in original_stat["stats"].keys():
                    if 'observation.images.' in key and key != main_camera_key:
                        additional_cameras.append(key)
                
                for camera in additional_cameras:
                    if video_mode == "multiview" and camera in original_stat["stats"]:
                        updated_stats_dict[camera] = original_stat["stats"][camera]
            
            updated_stat["stats"] = updated_stats_dict
            updated_stats.append(updated_stat)
        else:
            # 변환된 데이터에 없는 에피소드는 원본 그대로 유지
            updated_stats.append(original_stat)
    
    # 새로운 episodes_stats.jsonl 저장
    stats_file = meta_dir / "episodes_stats.jsonl"
    with open(stats_file, 'w') as f:
        for stat in updated_stats:
            f.write(json.dumps(stat) + '\n')
    
    logger.info(f"Updated episodes_stats.jsonl with {len(updated_stats)} episodes")

def main():
    parser = argparse.ArgumentParser(description="Convert dataset for pi0 model")
    parser.add_argument("--source_dir", type=str, 
                       default="/virtual_lab/rlwrld/david/.cache/huggingface/lerobot/RLWRLD/250716/allex_gesture_easy_all",
                       help="Source dataset directory")
    parser.add_argument("--task_name", type=str,
                       default="allex_gesture_easy_all",
                       help="Task name for directory structure")
    parser.add_argument("--state_mode", type=str, required=True,
                       choices=["pos_only", "pos_vel", "pos_vel_torq"],
                       help="State mode: pos_only, pos_vel, or pos_vel_torq")
    parser.add_argument("--action_mode", type=str, required=True,
                       choices=["right_arm", "dual_arm"],
                       help="Action mode: right_arm or dual_arm")
    parser.add_argument("--video_mode", type=str, required=True,
                       choices=["robotview", "multiview"],
                       help="Video mode: robotview (only robot0_robotview) or multiview (all cameras)")
    parser.add_argument("--state_indices", type=str, required=True,
                       help="Comma-separated state indices")
    parser.add_argument("--action_indices", type=str, required=True,
                       help="Comma-separated action indices")
    
    args = parser.parse_args()
    
    # Parse indices
    state_indices = [int(i) for i in args.state_indices.split(',')]
    action_indices = [int(i) for i in args.action_indices.split(',')]
    
    logger.info(f"State mode: {args.state_mode}, Action mode: {args.action_mode}, Video mode: {args.video_mode}")
    logger.info(f"State indices ({len(state_indices)}): {state_indices}")
    logger.info(f"Action indices ({len(action_indices)}): {action_indices}")
    
    # Setup directories
    source_base_dir = Path(args.source_dir)
    # source_dir의 한 단계 상위 폴더에 pi0_data 폴더 생성
    parent_dir = source_base_dir.parent
    
    # 폴더명을 짧은 약어로 변환
    def get_short_name(state_mode, action_mode, video_mode):
        # State mode 약어
        state_map = {
            "pos_only": "P",
            "pos_vel": "PV", 
            "pos_vel_torq": "PVT"
        }
        
        # Action mode 약어
        action_map = {
            "right_arm": "R",
            "dual_arm": "D"
        }
        
        # Video mode 약어
        video_map = {
            "robotview": "R",
            "multiview": "M"
        }
        
        return f"{state_map[state_mode]}_{action_map[action_mode]}_{video_map[video_mode]}"
    
    short_name = get_short_name(args.state_mode, args.action_mode, args.video_mode)
    target_base_dir = parent_dir / "pi0" / args.task_name / short_name
    
    source_data_dir = source_base_dir / "data"
    source_videos_dir = source_base_dir / "videos"
    target_data_dir = target_base_dir / "data"
    target_meta_dir = target_base_dir / "meta"
    target_videos_dir = target_base_dir / "videos"
    
    # 타겟 디렉토리 생성
    target_base_dir.mkdir(parents=True, exist_ok=True)
    
    if target_data_dir.exists():
        shutil.rmtree(target_data_dir)
    target_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 로봇 타입 감지
    robot_type = "franka" if "franka" in args.source_dir.lower() else "allex"
    logger.info(f"Detected robot type: {robot_type}")
    
    # 1. 데이터 변환
    logger.info("데이터 변환 시작...")
    
    for parquet_file in source_data_dir.glob("**/*.parquet"):
        try:
            logger.info(f"Processing {parquet_file.relative_to(source_data_dir)}")
            df = pd.read_parquet(parquet_file)
            df = transform_state_action(df, state_indices, action_indices, args.video_mode, robot_type)
            rel_path = parquet_file.relative_to(source_data_dir)
            target_file = target_data_dir / rel_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(target_file)
        except Exception as e:
            logger.error(f"Error processing {parquet_file}: {e}")
    
    # 2. meta 폴더 복사 및 수정
    source_meta_dir = source_base_dir / "meta"
    if source_meta_dir.exists():
        logger.info("meta 폴더 복사 및 수정 중...")
        if target_meta_dir.exists():
            shutil.rmtree(target_meta_dir)
        shutil.copytree(source_meta_dir, target_meta_dir)
        
        # info.json 수정
        info_file = target_meta_dir / "info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                info_data = json.load(f)
            
            # video_mode에 따른 이미지 정보 처리
            if args.video_mode == "robotview":
                # robotview만 사용하는 경우 추가 카메라들 제거
                additional_cameras = []
                
                # 로봇 타입에 따른 메인 카메라 키 결정
                if robot_type == "franka":
                    main_camera_key = "observation.images.agentview"
                else:
                    main_camera_key = "observation.images.robot0_robotview"
                
                # 최상위 레벨에서 메인 카메라를 제외한 모든 이미지 카메라 찾기
                for key in info_data.keys():
                    if 'observation.images.' in key and key != main_camera_key:
                        additional_cameras.append(key)
                
                # features 섹션에서도 확인
                if "features" in info_data:
                    for key in info_data["features"].keys():
                        if 'observation.images.' in key and key != main_camera_key:
                            additional_cameras.append(key)
                
                # 중복 제거
                additional_cameras = list(set(additional_cameras))
                
                for camera in additional_cameras:
                    # 최상위 레벨에서 제거
                    if camera in info_data:
                        del info_data[camera]
                        logger.info(f"Removed {camera} information from info.json root level")
                    
                    # features 섹션에서 제거
                    if "features" in info_data and camera in info_data["features"]:
                        del info_data["features"][camera]
                        logger.info(f"Removed {camera} information from features in info.json")
            # multiview인 경우 모든 카메라 정보 유지
            
            # observation.state shape과 names 수정
            if "observation.state" in info_data.get("features", {}):
                info_data["features"]["observation.state"]["shape"] = [len(state_indices)]
                
                # state names도 선택된 인덱스에 맞게 필터링
                original_state_names = info_data["features"]["observation.state"]["names"]
                if len(original_state_names) >= max(state_indices) + 1:
                    info_data["features"]["observation.state"]["names"] = [original_state_names[i] for i in state_indices]
                    logger.info(f"Updated observation.state shape to {len(state_indices)} dimensions and filtered names")
                else:
                    logger.warning(f"Original state names length ({len(original_state_names)}) is less than expected")
                    logger.info(f"Updated observation.state shape to {len(state_indices)} dimensions")
            
            # action shape 수정
            if "action" in info_data.get("features", {}):
                info_data["features"]["action"]["shape"] = [len(action_indices)]
                
                # action names 수정
                original_names = info_data["features"]["action"]["names"]
                if len(original_names) >= max(action_indices) + 1:
                    info_data["features"]["action"]["names"] = [original_names[i] for i in action_indices]
                    logger.info(f"Updated action names to {len(action_indices)} dimensions")
                else:
                    logger.warning(f"Original action names length ({len(original_names)}) is less than expected")
            
            # video_path 수정
            if "video_path" in info_data:
                if args.video_mode == "robotview":
                    # 로봇 타입에 따른 메인 카메라 키 결정
                    if robot_type == "franka":
                        main_camera_key = "observation.images.agentview"
                    else:
                        main_camera_key = "observation.images.robot0_robotview"
                    
                    info_data["video_path"] = f"videos/chunk-{{episode_chunk:03d}}/{main_camera_key}/episode_{{episode_index:06d}}.mp4"
                    logger.info(f"Updated video_path to use only {main_camera_key}")
                # multiview인 경우 기존 video_path 유지 (원본 그대로)
            
            # 수정된 info.json 저장
            with open(info_file, 'w') as f:
                json.dump(info_data, f, indent=4)
        
        # episodes_stats.jsonl 재생성
        logger.info("episodes_stats.jsonl 재생성 중...")
        generate_episodes_stats(target_data_dir, target_meta_dir, source_meta_dir, args.video_mode, robot_type)
    
    # 3. videos 폴더 복사 (video_mode에 따라)
    source_videos_dir = source_base_dir / "videos"
    if source_videos_dir.exists():
        if target_videos_dir.exists():
            shutil.rmtree(target_videos_dir)
        target_videos_dir.mkdir(parents=True, exist_ok=True)
        
        if args.video_mode == "robotview":
            # 로봇 타입에 따른 메인 카메라 키 결정
            if robot_type == "franka":
                main_camera_key = "observation.images.agentview"
            else:
                main_camera_key = "observation.images.robot0_robotview"
            
            logger.info(f"{main_camera_key} 비디오만 복사 중...")
            # chunk-000/observation.images.{main_camera_key} 폴더만 복사
            source_robotview_dir = source_videos_dir / "chunk-000" / main_camera_key
            target_robotview_dir = target_videos_dir / "chunk-000" / main_camera_key
            
            if source_robotview_dir.exists():
                target_robotview_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source_robotview_dir, target_robotview_dir)
                logger.info(f"Copied {len(list(source_robotview_dir.glob('*.mp4')))} {main_camera_key} videos")
            else:
                logger.warning(f"Source {main_camera_key} directory not found: {source_robotview_dir}")
        
        elif args.video_mode == "multiview":
            logger.info("모든 카메라 비디오 복사 중...")
            # 전체 videos 폴더 복사
            shutil.copytree(source_videos_dir, target_videos_dir, dirs_exist_ok=True)
            total_videos = len(list(target_videos_dir.glob("**/*.mp4")))
            logger.info(f"Copied {total_videos} videos from all cameras")
    
    logger.info(f"변환 완료! 결과 저장 위치: {target_base_dir}")

if __name__ == "__main__":
    main()