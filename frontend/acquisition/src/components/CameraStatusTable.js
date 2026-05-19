import React, { useContext, useState } from 'react';
import { Table, Button } from 'react-bootstrap';
import Accordion from 'react-bootstrap/Accordion';
import { AcquisitionState, isBusyPySpinState } from "../AcquisitionApi";

const RestoreDefaultsButton = ({ serial }) => {
    const { restoreCameraDefaults, recordingSystemStatus } = useContext(AcquisitionState);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState(null);
    const isPySpinBusy = isBusyPySpinState(recordingSystemStatus);

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
                disabled={busy || isPySpinBusy}
                size="sm"
                variant="outline-warning"
                title={isPySpinBusy
                    ? `Wait until the system is Idle (currently ${recordingSystemStatus})`
                    : 'Load factory UserSet on this camera and re-init'}
            >
                {busy ? 'Restoring…' : 'Restore defaults'}
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
                                    <td><RestoreDefaultsButton serial={cameraStatus.SerialNumber} /></td>
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
