/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Native Eigen3 evaluator for ConstantAdapter
 */

#ifndef OPENKALMAN_EIGEN_NATIVE_EVALUATORS_CONSTANTADAPTER_HPP
#define OPENKALMAN_EIGEN_NATIVE_EVALUATORS_CONSTANTADAPTER_HPP


namespace Eigen::internal
{
  template<typename PatternMatrix, typename Scalar, auto...constant>
  struct evaluator<OpenKalman::ConstantAdapter<PatternMatrix, Scalar, constant...>>
    : evaluator_base<OpenKalman::ConstantAdapter<PatternMatrix, Scalar, constant...>>
  {
    using XprType = OpenKalman::ConstantAdapter<PatternMatrix, Scalar, constant...>;
    using M = OpenKalman::Eigen3::eigen_matrix_t<Scalar, OpenKalman::index_dimension_of_v<PatternMatrix, 0>,
      OpenKalman::index_dimension_of_v<PatternMatrix, 1>>;

    enum {
      CoeffReadCost = 0,
      Flags = NoPreferredStorageOrderBit | LinearAccessBit | (traits<M>::Flags & RowMajorBit) |
        (packet_traits<Scalar>::Vectorizable ? PacketAccessBit : 0),
      Alignment = AlignedMax
    };

    explicit constexpr evaluator(const XprType& t) : m_xpr{t} {}

    constexpr Scalar coeff(Index row, Index col) const
    {
      return m_xpr.get_scalar_constant();
    }

    constexpr Scalar coeff(Index row) const
    {
      return m_xpr.get_scalar_constant();
    }

    template<int LoadMode, typename PacketType>
    constexpr PacketType packet(Index row, Index col) const
    {
      return internal::pset1<PacketType>(m_xpr.get_scalar_constant());
    }

    template<int LoadMode, typename PacketType>
    constexpr PacketType packet(Index row) const
    {
      return internal::pset1<PacketType>(m_xpr.get_scalar_constant());
    }

  protected:

    const XprType& m_xpr;
  };

} // namespace Eigen::internal

#endif //OPENKALMAN_EIGEN_NATIVE_EVALUATORS_CONSTANTADAPTER_HPP
