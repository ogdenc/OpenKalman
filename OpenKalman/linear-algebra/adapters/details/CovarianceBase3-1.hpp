/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definitions for CovarianceBase, case 3, subcase 1 (assignable NestedMatrix).
 */

#ifndef OPENKALMAN_COVARIANCEBASE3_1_HPP
#define OPENKALMAN_COVARIANCEBASE3_1_HPP

#include <utility>

namespace OpenKalman::internal
{
  // ======================================CASE 3======================================
  /**
   * \internal
   * \brief Base of Covariance and SquareRootCovariance classes (Case 3, subcase 1).
   * \details This specialization is operative if ArgType is self-contained and
   * # Derived is a square root or NestedMatrix is not self-adjoint;
   * # Derived is not a square root or NestedMatrix is not triangular; and
   * # NestedMatrix is assignable in that it is not constant and does not have any constant nested matrices.
   * In this case, NestedMatrix and the cholesky nested matrix are different.
   * \todo add an std::assignable_from constraint
   */
#ifdef __cpp_concepts
  template<typename Derived, typename NestedMatrix> requires
    (not case1or2<Derived, NestedMatrix>) and self_contained<NestedMatrix> and
    std::default_initializable<NestedMatrix> and
    std::assignable_from<std::add_lvalue_reference_t<NestedMatrix>, NestedMatrix>
  struct CovarianceBase<Derived, NestedMatrix>
#else
  template<typename Derived, typename NestedMatrix>
  struct CovarianceBase<Derived, NestedMatrix, std::enable_if_t<
    (not case1or2<Derived, NestedMatrix>) and self_contained<NestedMatrix> and
    stdcompat::default_initializable<NestedMatrix> and
    std::assignable_from<std::add_lvalue_reference_t<NestedMatrix>, NestedMatrix>>>
#endif
    : CovarianceBase3Impl<Derived, NestedMatrix>
  {
  private:

    using Base = CovarianceBase3Impl<Derived, NestedMatrix>;


#ifdef __cpp_concepts
    template<typename, typename>
#else
    template<typename, typename, typename>
#endif
    friend struct CovarianceBase;

    // The cholesky nested matrix one would expect given whether the covariance is a square root.
    using Base::cholesky_nested;

    // The synchronization state. 0 == synchronized, 1 = forward, -1 = reverse.
    using Base::synch_direction;

  protected:

    using typename Base::CholeskyNestedMatrix;

    static constexpr bool reverse_synchronizable = true;

    /**
     * \internal
     * \brief Synchronize from cholesky_nested_matrix to nested_object.
     */
    void synchronize_reverse() &
    {
      Base::nested_object() = to_covariance_nestable<NestedMatrix>(cholesky_nested);
      synch_direction = 0;
    }


    /// \internal \overload
    void synchronize_reverse() &&
    {
      Base::nested_object() = to_covariance_nestable<NestedMatrix>(std::move(cholesky_nested));
      synch_direction = 1;
    }


    /// \internal \overload
    constexpr void synchronize_reverse() const & {
      const_cast<CovarianceBase&>(*this).nested_object() = to_covariance_nestable<NestedMatrix>(cholesky_nested);
      synch_direction = 0;
    }


    /// \internal \overload
    constexpr void synchronize_reverse() const && {
      const_cast<CovarianceBase&>(*this).nested_object() =
        to_covariance_nestable<NestedMatrix>(std::move(cholesky_nested));
      synch_direction = 1;
    }

  public:

    /// Default constructor.
#ifdef __cpp_concepts
    CovarianceBase() requires self_contained<NestedMatrix> and std::default_initializable<NestedMatrix>
#else
    template<bool Enable = true, std::enable_if_t<Enable and
      self_contained<NestedMatrix> and stdcompat::default_initializable<NestedMatrix>, int> = 0>
    CovarianceBase()
#endif
      : Base {} {}


    /// Copy constructor.
    CovarianceBase(const CovarianceBase& other)
      : Base {other.synch_direction < 0 ? NestedMatrix {} : NestedMatrix {other.nested_object()},
              stdcompat::default_initializable<CholeskyNestedMatrix> and other.synch_direction > 0 ?
                CholeskyNestedMatrix {} : CholeskyNestedMatrix {other.cholesky_nested},
              other.synch_direction} {}


    /// Move constructor.
    CovarianceBase(CovarianceBase&& other) noexcept
      : Base {std::move(other).nested_object(), std::move(other).cholesky_nested, other.synch_direction} {}


    /**
     * \brief Construct from another \ref covariance.
     * \details This overload is operative if Arg's nested matrix and its cholesky nested matrix are the same.
     * In other words, Arg is a \ref CovarianceBaseCase3 "Case 3" or \ref CovarianceBaseCase4 "Case 4" covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires
      (not std::derived_from<std::decay_t<Arg>, CovarianceBase>) and (not case1or2<Arg>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      (not std::is_base_of_v<CovarianceBase, std::decay_t<Arg>>) and (not case1or2<Arg>), int> = 0>
#endif
    CovarianceBase(Arg&& arg)
      : Base {
          arg.synchronization_direction() < 0 ?
            NestedMatrix {} :
            NestedMatrix {to_covariance_nestable<NestedMatrix>(arg)},
          stdcompat::default_initializable<CholeskyNestedMatrix> and arg.synchronization_direction() > 0 ?
            CholeskyNestedMatrix {} :
            CholeskyNestedMatrix {to_covariance_nestable<CholeskyNestedMatrix>(std::forward<Arg>(arg))},
          arg.synchronization_direction()} {}


    /**
     * \brief Construct from another \ref covariance.
     * \details This overload is operative if Arg's nested matrix is the same as its cholesky nested matrix
     * as well as NestedMatrix. This only occurs when constructing the square or square_root of a matrix.
     * In other words, Arg is a \ref CovarianceBaseCase1 "Case 1" or \ref CovarianceBaseCase2 "Case 2" covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires case1or2<Arg> and
      (triangular_matrix<nested_object_of_t<Arg>> == triangular_matrix<NestedMatrix>) and (not diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and case1or2<Arg> and
      (triangular_matrix<nested_object_of_t<Arg>> == triangular_matrix<NestedMatrix>) and
      (not diagonal_matrix<Arg>), int> = 0>
#endif
    CovarianceBase(Arg&& arg)
      : Base {to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg)), 1} {}


    /**
     * \brief Construct from another \ref covariance.
     * \details This overload is operative if Arg's nested matrix is the same as its cholesky nested matrix
     * but not NestedMatrix.
     * In other words, Arg is a \ref CovarianceBaseCase1 "Case 1" or \ref CovarianceBaseCase2 "Case 2" covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires case1or2<Arg> and
      (triangular_matrix<nested_object_of_t<Arg>> != triangular_matrix<NestedMatrix> or diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and case1or2<Arg> and
      (triangular_matrix<nested_object_of_t<Arg>> != triangular_matrix<NestedMatrix> or diagonal_matrix<Arg>), int> = 0>
#endif
    CovarianceBase(Arg&& arg)
      : Base {to_covariance_nestable<CholeskyNestedMatrix>(std::forward<Arg>(arg)), -1} {}


    /**
     * \brief Construct from a \ref covariance_nestable of matching triangle/self-adjoint type.
     */
#ifdef __cpp_concepts
    template<covariance_nestable Arg> requires
      (triangular_matrix<Arg> == triangular_matrix<NestedMatrix>) and (not diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg> and
      (triangular_matrix<Arg> == triangular_matrix<NestedMatrix>) and (not diagonal_matrix<Arg>), int> = 0>
#endif
    explicit CovarianceBase(Arg&& arg)
      : Base {to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg)), 1} {}


    /**
     * \brief Construct from a \ref covariance_nestable of non-matching triangle/self-adjoint type.
     */
#ifdef __cpp_concepts
    template<covariance_nestable Arg> requires
      (triangular_matrix<Arg> != triangular_matrix<NestedMatrix> or diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg> and
      (triangular_matrix<Arg> != triangular_matrix<NestedMatrix> or diagonal_matrix<Arg>), int> = 0>
#endif
    explicit CovarianceBase(Arg&& arg)
      : Base {to_covariance_nestable<CholeskyNestedMatrix>(std::forward<Arg>(arg)), -1} {}


    using Base::operator=;


    /// Copy assignment operator.
    auto& operator=(const CovarianceBase& other)
    {
      return Base::operator=(other);
    }


    /// Move assignment operator.
    auto& operator=(CovarianceBase&& other) noexcept
    {
      return Base::operator=(std::move(other));
    }


  };


}

#endif
