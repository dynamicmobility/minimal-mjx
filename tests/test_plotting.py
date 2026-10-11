import numpy as np
import pytest

from minimal_mjx.utils import plotting


@pytest.mark.parametrize('n,expected', [(1, (1, 1)), (2, (1, 2)), (4, (2, 2)), (5, (2, 3)), (10, (3, 4))])
def test_get_subplot_grid(n, expected):
    nrows, ncols = plotting.get_subplot_grid(n)
    assert (nrows, ncols) == expected
    assert nrows * ncols >= n


def test_infer_frame_dim(np_env):
    g = np_env.mj_model.vis.global_
    assert plotting.infer_frame_dim(np_env.mj_model, None, None) == (g.offwidth, g.offheight)
    assert plotting.infer_frame_dim(np_env.mj_model, 10, 20) == (10, 20)


def test_reward_plotter_plot():
    rp = plotting.RewardPlotter(['a', 'b'])
    for i in range(5):
        rp.add_row({'a': i, 'b': -i}, reward=0.5 * i)
    assert rp.axdata == {'a': list(range(5)), 'b': [-i for i in range(5)]}
    fig, axs = rp.plot()
    assert axs[2].get_title() == 'Total Reward'
    plotting.plt.close(fig)


def test_info_plotter_h5_round_trip(tmp_path):
    ip = plotting.InfoPlotter(plotkey=['x'])
    for t in range(4):
        ip.add_row(0.1 * t, {'x': np.array([t, 2 * t])})
    path = tmp_path / 'info.h5'
    ip.save_to_h5(path)
    loaded = plotting.load_dict_from_hdf5(path)
    np.testing.assert_allclose(loaded['time'], [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_array_equal(loaded['x'], [[0, 0], [1, 2], [2, 4], [3, 6]])


def test_mujoco_plotter_h5_round_trip(tmp_path, np_env):
    mp = plotting.MujocoPlotter()
    state = np_env.reset(None)
    for _ in range(3):
        mp.add_row(state.data)
        state = np_env.step(state, np.zeros(np_env.action_size))
    path = tmp_path / 'data.h5'
    mp.save_to_h5(path)
    loaded = plotting.load_dict_from_hdf5(path)
    assert loaded['qpos'].shape == (3, np_env.nq)
    assert loaded['ctrl'].shape == (3, np_env.nu)


def test_save_metrics_writes_file(tmp_path):
    rp = plotting.RewardPlotter(['a'])
    rp.add_row({'a': 1.0}, 1.0)
    path = tmp_path / 'metrics.pdf'
    plotting.save_metrics(rp, path)
    assert path.exists()


def test_save_metrics_declined_dir(tmp_path, monkeypatch):
    monkeypatch.setattr('builtins.input', lambda *_: 'n')
    path = tmp_path / 'missing' / 'metrics.pdf'
    plotting.save_metrics(plotting.RewardPlotter([]), path)
    assert not path.parent.exists()


def test_check_directory_exists_rejects_file_path(tmp_path):
    with pytest.raises(Exception, match='not a directory'):
        plotting.check_directory_exists(tmp_path / 'file.txt')


@pytest.mark.slow
def test_save_video_writes_file(tmp_path):
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(5)]
    path = tmp_path / 'video.mp4'
    plotting.save_video(frames, 0.04, path)
    assert path.exists() and path.stat().st_size > 0


def test_progress_writes_csv_and_logs(tmp_path):
    logged = []

    class Run:
        def log(self, data, step):
            logged.append((data, step))

    metrics = {'eval/episode_reward': 1.0, 'eval/episode_reward_std': 0.1, 'note': 'skip'}
    progress = plotting.Progress(str(tmp_path), total_timesteps=100, run=Run())
    progress.plot_progress(10, metrics)
    assert (tmp_path / 'progress.csv').exists()
    assert logged == [({'eval/episode_reward': 1.0, 'eval/episode_reward_std': 0.1}, 10)]


def test_progress_load_resumes_curve(tmp_path):
    first = plotting.Progress(tmp_path)
    for step in (0, 10, 20):
        first.plot_progress(step, {'eval/episode_reward': step, 'eval/episode_reward_std': 0.0})
    resumed = plotting.Progress(tmp_path)
    resumed.load(before=20)
    assert resumed.x_data == [0, 10]


def test_progress_raises_on_nan(tmp_path):
    metrics = {'eval/episode_reward': float('nan'), 'eval/episode_reward_std': 0.0}
    with pytest.raises(Exception, match='NaN'):
        plotting.Progress(tmp_path, total_timesteps=1).plot_progress(0, metrics)
