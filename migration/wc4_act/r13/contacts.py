"""Raw contact and measured-motion diagnostics, not physical causality or scoring."""
import math


def diagnostic(body0,body1,raw_lambda,body_map,robot_ids,end_ids,door_ids,panel_id,
               angle_before,angle_after):
    if not math.isfinite(raw_lambda):raise ValueError('Nonfinite contact record')
    robot=body0 if body0 in robot_ids else body1 if body1 in robot_ids else None
    other=body1 if robot==body0 else body0
    pair_is_robot_door=robot is not None and other in door_ids
    end=pair_is_robot_door and robot in end_ids
    panel=end and other==panel_id
    nonzero=raw_lambda!=0
    decreasing=abs(angle_after)<abs(angle_before)
    nonend=pair_is_robot_door and robot not in end_ids
    return dict(robot_door_record=pair_is_robot_door,end_panel_raw_record=panel,
        nonzero_lambda_door_record=pair_is_robot_door and nonzero,
        zero_lambda_door_record=pair_is_robot_door and not nonzero,
        nonend_door_record=nonend,nonend_nonzero_door_record=nonend and nonzero,
        measured_angle_toward_closed=decreasing,
        nonend_nonzero_during_measured_closing=nonend and nonzero and decreasing,
        interpretation='Raw SDK record and coincident measured motion only; no calibrated force, work, causal pushing or failure conclusion',
        phase_used=False,automatic_failure=False)
