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
 * \brief Native Eigen3 evaluator for TriangularAdapter
 */

#ifndef OPENKALMAN_EIGEN_NATIVE_EVALUATORS_TRIANGULARADAPTER_HPP
#define OPENKALMAN_EIGEN_NATIVE_EVALUATORS_TRIANGULARADAPTER_HPP

namespace Eigen::internal
{
  template<typename ArgType, OpenKalman::triangle_type tri>
  struct evaluator<OpenKalman::TriangularAdapter<ArgType, tri>>
    : evaluator_base<OpenKalman::TriangularAdapter<ArgType, tri>>
  {
    using XprType = OpenKalman::TriangularAdapter<ArgType, tri>;
    using CoeffReturnType = typename std::decay_t<ArgType>::CoeffReturnType;
    using NestedEvaluator = evaluator<std::decay_t<ArgType>>;
    using Scalar = typename traits<std::decay_t<ArgType>>::Scalar;
    enum
    {
      CoeffReadCost = NestedEvaluator::CoeffReadCost,
      Flags = traits<XprType>::Flags,
      Alignment = NestedEvaluator::Alignment
    };

    //static constexpr bool is_row_major = static_cast<bool>(NestedEvaluator::Flags & RowMajorBit);


    explicit evaluator(const XprType& m_arg) : m_argImpl {m_arg.nested_object()} {}


    auto& coeffRef(Index row, Index col)
    {
      static Scalar dummy;
      if constexpr(tri == OpenKalman::triangle_type::diagonal) {if (row != col) { dummy = 0; return dummy; }}
      else if constexpr(tri == OpenKalman::triangle_type::upper) {if (row > col) { dummy = 0; return dummy; }}
      else if constexpr(tri == OpenKalman::triangle_type::lower) {if (row < col) { dummy = 0; return dummy; }}
      return m_argImpl.coeffRef(row, col);
    }


    auto& coeffRef(Index i)
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Linear (single index) element access by reference is only available for one-by-one TriangularAdapter");

      return m_argImpl.coeffRef(i);
    }


    CoeffReturnType coeff(Index row, Index col) const
    {
      if constexpr(std::is_lvalue_reference_v<CoeffReturnType>)
      {
        static std::remove_reference_t<CoeffReturnType> dummy = 0;
        if constexpr(tri == OpenKalman::triangle_type::diagonal) {if (row != col) return dummy;}
        else if constexpr(tri == OpenKalman::triangle_type::upper) {if (row > col) return dummy;}
        else if constexpr(tri == OpenKalman::triangle_type::lower) {if (row < col) return dummy;}
      }
      else
      {
        if constexpr(tri == OpenKalman::triangle_type::diagonal) {if (row != col) return Scalar(0);}
        else if constexpr(tri == OpenKalman::triangle_type::upper) {if (row > col) return Scalar(0);}
        else if constexpr(tri == OpenKalman::triangle_type::lower) {if (row < col) return Scalar(0);}
      }
      return m_argImpl.coeff(row, col);
    }


    CoeffReturnType coeff(Index i) const
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Linear (single index) element access is only available for one-by-one TriangularAdapter");

      return m_argImpl.coeff(i);
    }


    template<int LoadMode, typename PacketType>
    constexpr PacketType packet(Index row, Index col) const
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Packet access is only available for one-by-one TriangularAdapter");

      return m_argImpl.template packet<LoadMode, PacketType>(row, col);
    }


    template<int LoadMode, typename PacketType>
    constexpr PacketType packet(Index index) const
    {
      static_assert(OpenKalman::one_dimensional<ArgType>,
        "Linear packet access is only available for one-by-one TriangularAdapter");

      return m_argImpl.template packet<LoadMode, PacketType>(index);
    }

  protected:

    NestedEvaluator m_argImpl;

  };


}

#endif
