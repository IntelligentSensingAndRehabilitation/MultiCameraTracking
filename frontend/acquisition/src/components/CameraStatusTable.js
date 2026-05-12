import React, { useContext, useState } from 'react';
import { Table, Button } from 'react-bootstrap';
import Accordion from 'react-bootstrap/Accordion';
import { AcquisitionState } from "../AcquisitionApi";

const RestoreDefaultsButton = ({ serial }) => {
    const { restoreCameraDefaults, recordingSystemStatus } = useContext(AcquisitionState);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState(null);
    const isRecording = recordingSystemStatus === 'Recording';

    const handleClick = async () => {
        const ok = window.confirm(
            `Restore camera ${serial} to factory defaults? ` +
            `The acquisition system will reload to re-apply the current config.`
        );
        if (!ok) return;
        setBusy(true);
        setError(null);
        try {
            await restoreCameraDefaults(serial);
        } catch (e) {
            const msg = (e && e.response && e.response.data && e.response.data.detail) || e.message;
            setError(msg || 'Restore failed.');
        } finally {
            setBusy(false);
        }
    };

    return (
        <div>
            <Button
                onClick={handleClick}
                disabled={busy || isRecording}
                size="sm"
                variant="outline-warning"
                title={isRecording ? 'Stop the recording before restoring defaults' : 'Load factory UserSet on this camera and re-init'}
            >
                {busy ? 'Restoring…' : 'Restore defaults'}
            </Button>
            {error && <div className="text-danger small mt-1">{error}</div>}
        </div>
    );
};

const BumpedButton = ({ serial }) => {
    const { markCameraBumped } = useContext(AcquisitionState);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState(null);

    const handleClick = async () => {
        const ok = window.confirm(
            `Mark camera ${serial} as bumped? Subsequent trials will record under ` +
            `a new camera_config_hash, so DataJoint will treat them as a new ` +
            `calibration setup. You should capture a new calibration recording ` +
            `before the next trial.`
        );
        if (!ok) return;
        setBusy(true);
        setError(null);
        try {
            await markCameraBumped(serial);
        } catch (e) {
            const msg = (e && e.response && e.response.data && e.response.data.detail) || e.message;
            setError(msg || 'Mark bumped failed.');
        } finally {
            setBusy(false);
        }
    };

    return (
        <div>
            <Button
                onClick={handleClick}
                disabled={busy}
                size="sm"
                variant="outline-info"
                title="Force a new camera_config_hash for subsequent trials"
            >
                {busy ? '…' : 'Bumped'}
            </Button>
            {error && <div className="text-danger small mt-1">{error}</div>}
        </div>
    );
};

const CameraStatusTable = ({ api }) => {

    const { cameraStatusList } = useContext(AcquisitionState);

    return (
        <Accordion defaultActiveKey="0" className="g-4 p-2">
            <Accordion.Item eventKey="0">
                <Accordion.Header>Camera Statuses</Accordion.Header>
                <Accordion.Body>
                    <Table id="camera_status_table" striped bordered hover>
                        <thead>
                            <tr>
                                <th>Serial Number</th>
                                <th>Status</th>
                                <th>Pixel Format</th>
                                <th>Binning Horizontal</th>
                                <th>Binning Vertical</th>
                                <th>Width</th>
                                <th>Height</th>
                                <th>Sync Offset</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {cameraStatusList.map((cameraStatus) => (
                                <tr key={cameraStatus.SerialNumber}>
                                    <td>{cameraStatus.SerialNumber}</td>
                                    <td>{cameraStatus.Status}</td>
                                    <td>{cameraStatus.PixelFormat}</td>
                                    <td>{cameraStatus.BinningHorizontal}</td>
                                    <td>{cameraStatus.BinningVertical}</td>
                                    <td>{cameraStatus.Width}</td>
                                    <td>{cameraStatus.Height}</td>
                                    <td>{cameraStatus.SyncOffset}</td>
                                    <td>
                                        <div className="d-flex gap-2">
                                            <RestoreDefaultsButton serial={cameraStatus.SerialNumber} />
                                            <BumpedButton serial={cameraStatus.SerialNumber} />
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </Table>
                </Accordion.Body>
            </Accordion.Item>
        </Accordion>

    );
};

export default CameraStatusTable;
