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
 * \brief Common class member functions for CovarianceBase, case 3.
 */

#ifndef OPENKALMAN_COVARIANCEBASE3IMPL_HPP
#define OPENKALMAN_COVARIANCEBASE3IMPL_HPP

#include <utility>

namespace OpenKalman::internal
{
  // ======================================CASE 3======================================
  /**
   * \internal
   * \anchor CovarianceBaseCase3
   * \brief Base of Covariance and SquareRootCovariance classes (Case 3).
   * \details This specialization is operative if ArgType is self-contained and
   * # Derived is a square root or NestedMatrix is not self-adjoint; and
   * # Derived is not a square root or NestedMatrix is not triangular.
   * In this case, NestedMatrix and the cholesky nested matrix are different.
   */
  template<typename Derived, typename NestedMatrix>
  struct CovarianceBase3Impl : MatrixBase<Derived, NestedMatrix>
  {
  private:

    using Base = MatrixBase<Derived, NestedMatrix>;

    using Scalar = scalar_type_of_t<NestedMatrix>;

  protected:

    using CholeskyNestedMatrix = std::conditional_t<
      diagonal_matrix<NestedMatrix>,
      typename MatrixTraits<NestedMatrix>::template DiagonalMatrixFrom<>,
      std::conditional_t<triangular_matrix<NestedMatrix>,
        typename MatrixTraits<NestedMatrix>::template SelfAdjointMatrixFrom<>,
        typename MatrixTraits<NestedMatrix>::template TriangularMatrixFrom<>>>;


    // The cholesky nested matrix one would expect given whether the covariance is a square root.
    mutable CholeskyNestedMatrix cholesky_nested;


    // The synchronization state. 0 == synchronized, 1 = forward, -1 = reverse.
    mutable int synch_direction;


    /**
     * \internal
     * \brief The cholesky nested matrix: self-adjoint if Covariance, triangular if SquareRootCovariance
     */
    auto& cholesky_nested_matrix() & { return cholesky_nested; }

    /// \overload
    const auto& cholesky_nested_matrix() const & { return cholesky_nested; }

    /// \overload
    auto&& cholesky_nested_matrix() && { return std::move(cholesky_nested); }

    /// \overload
    const auto&& cholesky_nested_matrix() const && { return std::move(cholesky_nested); }


    /**
     * \internal
     * \brief The synchronization direction. 0 == synchronized, 1 = forward, -1 = reverse.
     */
    int synchronization_direction() const { return synch_direction; }


    /**
     * \internal
     * \brief Synchronize from nested_matrix to cholesky_nested_matrix.
     */
    void synchronize_forward() const &
    {
      cholesky_nested = to_covariance_nestable<CholeskyNestedMatrix>(Base::nested_matrix());
      synch_direction = 0;
    }


    /// \internal \overload
    void synchronize_forward() const &&
    {
      cholesky_nested = to_covariance_nestable<CholeskyNestedMatrix>(std::move(*this).nested_matrix());
      synch_direction = -1;
    }


    /**
     * \internal
     * \brief Indicate that the nested matrix has changed.
     */
    void mark_nested_matrix_changed() const { synch_direction = 1; }


    /**
     * \internal
     * \brief Indicate that the cholesky nested matrix has changed.
     */
    void mark_cholesky_nested_matrix_changed() const { synch_direction = -1; }


    /**
     * \internal
     * \brief Indicate that the covariance is synchronized.
     */
    void mark_synchronized() const { synch_direction = 0; }


    /// Default constructor.
#ifdef __cpp_concepts
    CovarianceBase3Impl() requires std::default_initializable<NestedMatrix>
#else
    template<typename T = NestedMatrix, std::enable_if_t<std::is_default_constructible_v<T>, int> = 0>
    CovarianceBase3Impl()
#endif
      : Base {}, synch_direction {} {}


    /// Copy constructor.
    CovarianceBase3Impl(const CovarianceBase3Impl& other) = default;


    /// Move constructor.
    CovarianceBase3Impl(CovarianceBase3Impl&& other) noexcept = default;


    /**
     * \internal
     * \brief General constructor, setting nested_matrix and leaving cholesky_nested unset.
     * \tparam Arg A \ref OpenKalman::covariance_nestable "covariance_nestable"
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (triangular_matrix<Arg> == triangular_matrix<NestedMatrix>) and
      (not diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<(triangular_matrix<Arg> == triangular_matrix<NestedMatrix>) and
      (not diagonal_matrix<Arg>), int> = 0>
#endif
    CovarianceBase3Impl(Arg&& arg, int sd) noexcept
      : Base {std::forward<Arg>(arg)}, cholesky_nested {}, synch_direction {sd} {}


    /**
     * \internal
     * \brief General constructor, setting cholesky_nested and leaving nested_matrix unset.
     * \tparam Arg A \ref OpenKalman::covariance_nestable "covariance_nestable"
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (triangular_matrix<Arg> != triangular_matrix<NestedMatrix>) or diagonal_matrix<Arg>
#else
    template<typename Arg, std::enable_if_t<(triangular_matrix<Arg> != triangular_matrix<NestedMatrix>) or
      diagonal_matrix<Arg>, int> = 0>
#endif
    CovarianceBase3Impl(Arg&& arg, int sd) noexcept
      : Base {}, cholesky_nested {std::forward<Arg>(arg)}, synch_direction {sd} {}


    /**
     * \internal
     * \brief General constructor, setting both nested_matrix and cholesky_nested.
     * \tparam N A \ref OpenKalman::covariance_nestable "covariance_nestable" for the nested matrix
     * \tparam CN A \ref OpenKalman::covariance_nestable "covariance_nestable" for the cholesky nested matrix
     */
    template<typename N, typename CN>
    CovarianceBase3Impl(N&& n, CN&& cn, int sd) noexcept
      : Base {std::forward<N>(n)}, cholesky_nested {std::forward<CN>(cn)}, synch_direction {sd} {}


  public:

    /// Copy assignment operator.
    auto& operator=(const CovarianceBase3Impl& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        if (synch_direction >= 0) Base::nested_matrix() = other.nested_matrix();
        if (synch_direction <= 0) cholesky_nested = other.cholesky_nested_matrix();
        synch_direction = other.synch_direction;
      }
      return *this;
    }


    /// Move assignment operator.
    auto& operator=(CovarianceBase3Impl&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        if (synch_direction >= 0) Base::nested_matrix() = std::move(other.nested_matrix());
        if (synch_direction <= 0) cholesky_nested = std::move(other.cholesky_nested_matrix());
        synch_direction = other.synch_direction;
      }
      return *this;
    }


    /**
     * \internal
     * \brief Assign from another \ref covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires (not std::derived_from<std::decay_t<Arg>, CovarianceBase3Impl>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      (not std::is_base_of_v<CovarianceBase3Impl, std::decay_t<Arg>>), int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr(not (zero_matrix<nested_matrix_of_t<Arg>> and zero_matrix<NestedMatrix>) and
        not (identity_matrix<nested_matrix_of_t<Arg>> and identity_matrix<NestedMatrix>))
      {
        if constexpr (case1or2<Arg>)
        {
          // Arg is Case 1 or 2
          if constexpr (triangular_matrix<nested_matrix_of_t<Arg>> == triangular_matrix<NestedMatrix> and
            not diagonal_matrix<Arg>)
          {
            Base::nested_matrix() = to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg));
            mark_nested_matrix_changed();
          }
          else
          {
            cholesky_nested = to_covariance_nestable<CholeskyNestedMatrix>(std::forward<Arg>(arg));
            mark_cholesky_nested_matrix_changed();
          }
        }
        else
        {
          // Arg is Case 3 or 4
          if (synch_direction >= 0)
          {
            Base::nested_matrix() = to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg));
          }
          if (synch_direction <= 0)
          {
            cholesky_nested = to_covariance_nestable<CholeskyNestedMatrix>(std::forward<Arg>(arg));
          }
          synch_direction = arg.synchronization_direction();
        }
      }
      return *this;
    }


    /**
     * \brief Assign from a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<covariance_nestable Arg>
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr(not (zero_matrix<Arg> and zero_matrix<NestedMatrix>) and
        not (identity_matrix<Arg> and identity_matrix<NestedMatrix>))
      {
        if constexpr(triangular_matrix<Arg> == triangular_matrix<NestedMatrix> and not diagonal_matrix<Arg>)
        {
          Base::nested_matrix() = to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg));
          mark_nested_matrix_changed();
        }
        else
        {
          cholesky_nested = to_covariance_nestable<CholeskyNestedMatrix>(std::forward<Arg>(arg));
          mark_cholesky_nested_matrix_changed();
        }
      }
      return *this;
    }


    /**
     * \brief Get or set element (i, j) of the covariance matrix.
     * \param i The row.
     * \param j The column.
     * \return An ElementAccessor object.
     */
    auto operator() (std::size_t i, std::size_t j)
    {
      if constexpr(element_settable<CholeskyNestedMatrix, std::size_t, std::size_t>)
        return ElementAccessor(cholesky_nested, i, j,
          [this] { if (synch_direction > 0) synchronize_forward(); },
          [this] { mark_cholesky_nested_matrix_changed(); });
      else
        return ElementAccessor(cholesky_nested_matrix(), i, j,
          [this] { if (synch_direction > 0) synchronize_forward(); });
    }

    /// \overload
    auto operator() (std::size_t i, std::size_t j) const
    {
      return ElementAccessor(cholesky_nested, i, j, [this] { if (synch_direction > 0) synchronize_forward(); });
    }


    /**
     * \brief Get or set element i of the covariance matrix.
     * \param i The row.
     * \return An ElementAccessor object.
     */
    auto operator[] (std::size_t i)
    {
      if constexpr(element_settable<CholeskyNestedMatrix, std::size_t>)
        return ElementAccessor(cholesky_nested, i,
          [this] { if (synch_direction > 0) synchronize_forward(); },
          [this] { mark_cholesky_nested_matrix_changed(); });
      else
        return ElementAccessor(cholesky_nested, i,
          [this] { if (synch_direction > 0) synchronize_forward(); });
    }


    /// \overload
    auto operator[] (std::size_t i) const
    {
      return ElementAccessor(cholesky_nested, i, [this] { if (synch_direction > 0) synchronize_forward(); });
    }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
    void set_element(const Scalar s, const std::size_t i, const std::size_t j)
    {
      if (synch_direction > 0) synchronize_forward();
      OpenKalman::set_element(cholesky_nested, s, i, j);
      mark_cholesky_nested_matrix_changed();
    }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
    void set_element(const Scalar s, const std::size_t i)
    {
      if (synch_direction > 0) synchronize_forward();
      OpenKalman::set_element(cholesky_nested, s, i);
      mark_cholesky_nested_matrix_changed();
    }

  };


}

#endif //OPENKALMAN_COVARIANCEBASE3IMPL_HPP
