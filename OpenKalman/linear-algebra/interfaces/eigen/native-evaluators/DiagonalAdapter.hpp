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
 * \brief Native Eigen3 evaluator for diagonal_adapter
 */

#ifndef OPENKALMAN_EIGEN_NATIVE_EVALUATORS_DIAGONALADAPTER_HPP
#define OPENKALMAN_EIGEN_NATIVE_EVALUATORS_DIAGONALADAPTER_HPP

namespace Eigen::internal
{
  template<typename ArgType>
  struct evaluator<OpenKalman::diagonal_adapter<ArgType>>
    : evaluator_base<OpenKalman::diagonal_adapter<ArgType>>
  {
    using XprType = OpenKalman::diagonal_adapter<ArgType>;
    using CoeffReturnType = typename std::decay_t<ArgType>::CoeffReturnType;
    using NestedEvaluator = evaluator<std::decay_t<ArgType>>;
    using Scalar = typename traits<std::decay_t<ArgType>>::Scalar;
    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost,
      Flags = NestedEvaluator::Flags & (~DirectAccessBit) & (~LinearAccessBit) & (~PacketAccessBit),
      Alignment = NestedEvaluator::Alignment
    };

    explicit evaluator(const XprType& m_arg) : m_argImpl(m_arg.nested_object()) {}

    auto& coeffRef(Index row, Index col)
    {
      if (row == col)
        return m_argImpl.coeffRef(row);
      else
      {
        static Scalar dummy;
        dummy = 0;
        return dummy;
      }
    }

    auto& coeffRef(Index index)
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Linear (single index) element access by reference is only available for one-by-one diagonal_adapter");

      return m_argImpl.coeffRef(index);
    }

    CoeffReturnType coeff(Index row, Index col) const
    {
      if (row == col)
        return m_argImpl.coeff(row);
      else
      {
        if constexpr(std::is_lvalue_reference_v<CoeffReturnType>)
        {
          static std::remove_reference_t<CoeffReturnType> dummy = 0;
          return dummy;
        }
        else
          return Scalar(0);
      }
    }

    CoeffReturnType coeff(Index index) const
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Linear (single index) element access is only available for one-by-one diagonal_adapter");

      return m_argImpl.coeff(index);
    }


    template<int LoadMode, typename PacketType>
    constexpr PacketType packet(Index row, Index col) const
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Packet access is only available for one-by-one diagonal_adapter");

      return m_argImpl.template packet<LoadMode, PacketType>(row, col);
    }


    template<int LoadMode, typename PacketType>
    constexpr PacketType packet(Index index) const
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Linear packet access is only available for one-by-one diagonal_adapter");

      return m_argImpl.template packet<LoadMode, PacketType>(index);
    }

  protected:

    NestedEvaluator m_argImpl;
  };


}

#endif
