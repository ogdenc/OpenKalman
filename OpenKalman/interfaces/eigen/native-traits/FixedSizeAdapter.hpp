/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Native Eigen3 traits for Eigen3 general FixedSizeAdapter
 */

#ifndef OPENKALMAN_EIGEN_NATIVE_TRAITS_FIXEDSIZEADAPTER_HPP
#define OPENKALMAN_EIGEN_NATIVE_TRAITS_FIXEDSIZEADAPTER_HPP

namespace OpenKalman::Eigen3::internal
{
#ifdef __cpp_concepts
  template<OpenKalman::Eigen3::eigen_general NestedMatrix>
  struct native_traits<OpenKalman::internal::FixedSizeAdapter<NestedMatrix>>
#else
  template<typename NestedMatrix>
  struct native_traits<OpenKalman::internal::FixedSizeAdapter<NestedMatrix>, std::enable_if_t<OpenKalman::Eigen3::eigen_general<NestedMatrix>>>
#endif
    : Eigen::internal::traits<std::decay_t<NestedMatrix>>
  {
    enum
    {
      RowsAtCompileTime = 1,
      ColsAtCompileTime = 1,
      MaxRowsAtCompileTime = RowsAtCompileTime,
      MaxColsAtCompileTime = ColsAtCompileTime,
    };
  };


#ifdef __cpp_concepts
  template<OpenKalman::Eigen3::eigen_general NestedMatrix, typename Rows>
  struct native_traits<OpenKalman::internal::FixedSizeAdapter<NestedMatrix, Rows>>
#else
  template<typename NestedMatrix, typename Rows>
  struct native_traits<OpenKalman::internal::FixedSizeAdapter<NestedMatrix, Rows>, std::enable_if_t<
    OpenKalman::Eigen3::eigen_general<NestedMatrix>>>
#endif
    : Eigen::internal::traits<std::decay_t<NestedMatrix>>
  {
  private:

    static constexpr auto rows = OpenKalman::fixed_vector_space_descriptor<Rows> ? static_cast<int>(OpenKalman::dimension_size_of_v<Rows>) : Eigen::Dynamic;
    static constexpr auto row_major_bit = rows != 1 ? 0x0 : (Eigen::internal::traits<std::decay_t<NestedMatrix>>::Flags & Eigen::RowMajorBit);

  public:

    enum
    {
      RowsAtCompileTime = rows,
      ColsAtCompileTime = 1,
      MaxRowsAtCompileTime = RowsAtCompileTime,
      MaxColsAtCompileTime = ColsAtCompileTime,
      Flags = (Eigen::internal::traits<std::decay_t<NestedMatrix>>::Flags & ~Eigen::RowMajorBit) | row_major_bit,
    };
  };


#ifdef __cpp_concepts
  template<OpenKalman::Eigen3::eigen_general NestedMatrix, typename Rows, typename Cols>
  struct native_traits<OpenKalman::internal::FixedSizeAdapter<NestedMatrix, Rows, Cols>>
#else
  template<typename NestedMatrix, typename Rows, typename Cols>
  struct native_traits<OpenKalman::internal::FixedSizeAdapter<NestedMatrix, Rows, Cols>, std::enable_if_t<
    OpenKalman::Eigen3::eigen_general<NestedMatrix>>>
#endif
    : Eigen::internal::traits<std::decay_t<NestedMatrix>>
  {
  private:

    static constexpr auto rows = OpenKalman::fixed_vector_space_descriptor<Rows> ? static_cast<int>(OpenKalman::dimension_size_of_v<Rows>) : Eigen::Dynamic;
    static constexpr auto cols = OpenKalman::fixed_vector_space_descriptor<Cols> ? static_cast<int>(OpenKalman::dimension_size_of_v<Cols>) : Eigen::Dynamic;
    static constexpr auto row_major_bit =
      rows == 1 and cols != 1 ? Eigen::RowMajorBit :
      rows != 1 and cols == 1 ? 0x0 :
      (Eigen::internal::traits<std::decay_t<NestedMatrix>>::Flags & Eigen::RowMajorBit);

  public:

    enum
    {
      RowsAtCompileTime = rows,
      ColsAtCompileTime = cols,
      MaxRowsAtCompileTime = RowsAtCompileTime,
      MaxColsAtCompileTime = ColsAtCompileTime,
      Flags = (Eigen::internal::traits<std::decay_t<NestedMatrix>>::Flags & ~Eigen::RowMajorBit) | row_major_bit,
    };
  };

} // namespace OpenKalman::Eigen3::internal


#endif //OPENKALMAN_EIGEN_NATIVE_TRAITS_FIXEDSIZEADAPTER_HPP
