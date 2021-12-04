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
 * \brief Definitions for CovarianceBase, case 1.
 */

#ifndef OPENKALMAN_COVARIANCEBASE1_HPP
#define OPENKALMAN_COVARIANCEBASE1_HPP

namespace OpenKalman::internal
{

  /**
   * \internal
   * \brief Covariance Cov's cholesky nested matrix and nested matrix Nested are both either triangular or self-adjoint.
   */
  template<typename Cov, typename Nested = nested_matrix_t<Cov>>
#ifdef __cpp_concepts
  concept case1or2 = (self_adjoint_covariance<Cov> and self_adjoint_matrix<Nested>) or
    (triangular_covariance<Cov> and triangular_matrix<Nested>);
#else
  static constexpr bool case1or2 = (self_adjoint_covariance<Cov> and self_adjoint_matrix<Nested>) or
    (triangular_covariance<Cov> and triangular_matrix<Nested>);
#endif


  // =====================================CASE 1=======================================
  /**
   * \internal
   * \anchor CovarianceBaseCase1
   * \brief Base of Covariance and SquareRootCovariance classes (Case 1).
   * \details This specialization is operative if NestedMatrix is self-contained and either
   * # Derived is not a square root and NestedMatrix is self-adjoint; or
   * # Derived is a square root and NestedMatrix is triangular.
   * In this case, NestedMatrix and the cholesky nested matrix are the same.
   */
#ifdef __cpp_concepts
  template<typename Derived, typename NestedMatrix> requires
    case1or2<Derived, NestedMatrix> and self_contained<NestedMatrix>
  struct CovarianceBase<Derived, NestedMatrix>
#else
  template<typename Derived, typename NestedMatrix>
  struct CovarianceBase<Derived, NestedMatrix, std::enable_if_t<
    case1or2<Derived, NestedMatrix> and self_contained<NestedMatrix>>>
#endif
    : MatrixBase<Derived, NestedMatrix>
  {
  private:

    using Base = MatrixBase<Derived, NestedMatrix>;

    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;


#ifdef __cpp_concepts
    template<typename, typename>
#else
    template<typename, typename, typename>
#endif
    friend struct CovarianceBase;

  protected:

    using CholeskyNestedMatrix = std::conditional_t<diagonal_matrix<NestedMatrix>,
      typename MatrixTraits<NestedMatrix>::template DiagonalMatrixFrom<>,
      std::conditional_t<triangular_matrix<NestedMatrix>,
        typename MatrixTraits<NestedMatrix>::template SelfAdjointMatrixFrom<>,
        typename MatrixTraits<NestedMatrix>::template TriangularMatrixFrom<>>>;


    /**
     * \internal
     * \brief The cholesky factor or square of the nested matrix.
     * \details This will be triangular if Covariance or self-adjoint if SquareRootCovariance
     */
    decltype(auto) cholesky_nested_matrix() &
    {
      if constexpr (triangular_covariance<Derived>) return Cholesky_square(this->nested_matrix());
      else return Cholesky_factor(this->nested_matrix());
    }


    /// \internal \overload
    decltype(auto) cholesky_nested_matrix() const &
    {
      if constexpr (triangular_covariance<Derived>) return Cholesky_square(this->nested_matrix());
      else return Cholesky_factor(this->nested_matrix());
    }


    /// \internal \overload
    decltype(auto) cholesky_nested_matrix() &&
    {
      if constexpr (triangular_covariance<Derived>) return Cholesky_square(std::move(*this).nested_matrix());
      else return Cholesky_factor(std::move(*this).nested_matrix());
    }


    /// \internal \overload
    decltype(auto) cholesky_nested_matrix() const &&
    {
      if constexpr (triangular_covariance<Derived>) return Cholesky_square(std::move(*this).nested_matrix());
      else return Cholesky_factor(std::move(*this).nested_matrix());
    }


    /**
     * \internal
     * \brief The synchronization direction. 0 == synchronized, 1 = forward, -1 = reverse.
     */
    constexpr static int synchronization_direction() { return 0; }


    /**
     * \internal
     * \brief Synchronize the state from nested_matrix to cholesky_nested_matrix.
     * \details This is a no-op.
     */
    constexpr static void synchronize_forward() {};


    /**
     * \internal
     * \brief Synchronize the state from cholesky_nested_matrix to nested_matrix.
     * \details This is a no-op.
     */
    constexpr static void synchronize_reverse() {};


    /**
     * \internal
     * \brief Indicate that the nested matrix has changed.
     * \details In this specialization of CovarianceBase, this function is a no-op.
     */
    constexpr static void mark_nested_matrix_changed() {};


    /**
     * \internal
     * \brief Indicate that the cholesky nested matrix has changed.
     * \details In this specialization of CovarianceBase, this function is a no-op.
     */
    constexpr static void mark_cholesky_nested_matrix_changed() {};


    /**
     * \internal
     * \brief Indicate that the covariance is synchronized.
     * \details In this specialization of CovarianceBase, this function is a no-op.
     */
    constexpr static void mark_synchronized() {};


  public:
    /// Default constructor.
#ifdef __cpp_concepts
    CovarianceBase() requires std::default_initializable<NestedMatrix>
#else
    template<typename T = NestedMatrix, std::enable_if_t<std::is_default_constructible_v<T>, int> = 0>
    CovarianceBase()
#endif
      : Base {} {}


    /**
     * \brief Construct from another \ref covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires (not std::derived_from<std::decay_t<Arg>, CovarianceBase>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      (not std::is_base_of_v<CovarianceBase, std::decay_t<Arg>>), int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept : Base {to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg))} {}


    /**
     * \brief Construct from a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<covariance_nestable Arg>
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg>, int> = 0>
#endif
    explicit CovarianceBase(Arg&& arg) noexcept : Base {to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg))} {}


    /**
     * \internal
     * \brief Assign from another \ref covariance or a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (covariance<Arg> or covariance_nestable<Arg>) and
      (not std::derived_from<std::decay_t<Arg>, CovarianceBase>)
#else
    template<typename Arg, std::enable_if_t<(covariance<Arg> or covariance_nestable<Arg>) and
      (not std::is_base_of_v<CovarianceBase, std::decay_t<Arg>>), int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      Base::operator=(to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg)));
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
      return ElementAccessor(Base::nested_matrix(), i, j);
    }


    /// \overload
    auto operator() (std::size_t i, std::size_t j) const
    {
      return ElementAccessor(Base::nested_matrix(), i, j);
    }


    /**
     * \brief Get or set element i of the covariance matrix, if it is a vector.
     * \param i The row.
     * \return An ElementAccessor object.
     */
    auto operator[] (std::size_t i)
    {
      return ElementAccessor(Base::nested_matrix(), i);
    }


    /// \overload
    auto operator[] (std::size_t i) const
    {
      return ElementAccessor(Base::nested_matrix(), i);
    }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
    void set_element(const Scalar s, const std::size_t i, const std::size_t j)
    {
      OpenKalman::set_element(Base::nested_matrix(), s, i, j);
    }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
    void set_element(const Scalar s, const std::size_t i)
    {
      OpenKalman::set_element(Base::nested_matrix(), s, i);
    }

  };


}

#endif //OPENKALMAN_COVARIANCEBASE1_HPP
